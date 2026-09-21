"""Canonical chat rendering and content-only token selection (no model imports).

The diagnostic markers locate the structural insertion of Q. They never enter
the encoded model input. Offsets always come from the complete canonical render.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256

SCHEMA_VERSION = 1
ACTIVATION_LABELS = {
    "lora": "conditioned_context",
    "orig": "unconditioned_context",
    "diff": "prompt_difference",
}


def normalize_conditioning(conditioning: dict | None) -> dict | None:
    if conditioning is None:
        return None
    if not isinstance(conditioning, dict):
        raise ValueError("conditioning must be a mapping")
    if set(conditioning) - {"channel", "instruction", "instruction_sha256"}:
        raise ValueError("Unknown conditioning fields")
    if conditioning.get("channel") not in {"prefix", "system"}:
        raise ValueError("conditioning.channel must be prefix or system")
    instruction = conditioning.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("conditioning.instruction must be nonempty text")
    instruction = instruction.strip()
    digest = sha256(instruction.encode("utf-8")).hexdigest()
    if conditioning.get("instruction_sha256", digest) != digest:
        raise ValueError("conditioning.instruction_sha256 does not match instruction")
    return {
        "channel": conditioning["channel"],
        "instruction": instruction,
        "instruction_sha256": digest,
    }


def build_messages(context: str, conditioning: dict | None = None) -> list[dict]:
    conditioning = normalize_conditioning(conditioning)
    if conditioning is None:
        return [{"role": "user", "content": context}]
    instruction = conditioning["instruction"]
    if conditioning["channel"] == "prefix":
        return [{"role": "user", "content": instruction + "\n\n" + context}]
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": context},
    ]


@dataclass
class ContextInput:
    messages: list[dict]
    rendered: str
    input_ids: list[int]
    context_span: tuple[int, int]
    context_positions: list[int]
    context_token_ids: list[int]
    context_offsets: list[tuple[int, int]]
    boundary_tokens: list[dict]

    def diagnostics(self) -> dict:
        return asdict(self)


@dataclass
class PreparedContext:
    display_context: str
    target: ContextInput
    reference: ContextInput

    @property
    def aligned(self) -> bool:
        return (
            self.target.context_token_ids == self.reference.context_token_ids
            and self.target.context_offsets == self.reference.context_offsets
        )

    def require_alignment(self):
        if not self.aligned:
            raise ValueError(
                "Q token IDs/character spans differ; prompt subtraction is blocked"
            )


def prepare_input(
    tokenizer,
    context: str,
    conditioning: dict | None = None,
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    max_length: int | None = None,
) -> ContextInput:
    if not isinstance(context, str) or not context:
        raise ValueError("Q must be nonempty text")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "Content selection requires a fast tokenizer with character offsets"
        )
    conditioning = normalize_conditioning(conditioning)
    messages = build_messages(context, conditioning)
    render_args = dict(
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    rendered = tokenizer.apply_chat_template(messages, **render_args)

    # Choose markers absent from BOTH I and Q; searching Q itself is ambiguous.
    digest = sha256(rendered.encode("utf-8")).hexdigest()
    begin, end = f"__AO_Q_BEGIN_{digest}__", f"__AO_Q_END_{digest}__"
    if begin in rendered or end in rendered:
        raise ValueError("Diagnostic marker collision")
    marked = tokenizer.apply_chat_template(
        build_messages(begin + context + end, conditioning), **render_args
    )
    if marked.count(begin) != 1 or marked.count(end) != 1:
        raise ValueError("Chat template does not preserve a unique context insertion")
    start = marked.index(begin)
    end_start = marked.index(end)
    if marked[start + len(begin) : end_start] != context:
        raise ValueError("Chat template transforms Q; exact context cannot be selected")
    if marked[:start] + context + marked[end_start + len(end) :] != rendered:
        raise ValueError(
            "Diagnostic render does not reproduce canonical render; Q may be trimmed"
        )
    stop = start + len(context)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if len(ids) != len(offsets):
        raise ValueError("Tokenizer returned inconsistent offsets")
    limits = [
        n
        for n in (max_length, getattr(tokenizer, "model_max_length", None))
        if isinstance(n, int) and n > 0
    ]
    if limits and len(ids) > min(limits):
        raise ValueError(
            f"Full input has {len(ids)} tokens, exceeding limit {min(limits)}; no truncation allowed"
        )
    selected, relative, boundary = [], [], []
    for i, (left, right) in enumerate(offsets):
        if not 0 <= left <= right <= len(rendered):
            raise ValueError("Tokenizer offsets are not valid character coordinates")
        if start <= left < right <= stop:
            selected.append(i)
            relative.append((left - start, right - start))
        elif left < stop and right > start and left < right:
            boundary.append(
                {
                    "position": i,
                    "token_id": ids[i],
                    "offset": (left - start, right - start),
                }
            )
    if not selected:
        raise ValueError("No tokens wholly inside Q")
    return ContextInput(
        messages,
        rendered,
        ids,
        (start, stop),
        selected,
        [ids[i] for i in selected],
        relative,
        boundary,
    )


def prepare_context(
    tokenizer, context: str, conditioning: dict | None = None, **kwargs
) -> PreparedContext:
    reference = prepare_input(tokenizer, context, **kwargs)
    target = (
        prepare_input(tokenizer, context, conditioning, **kwargs)
        if conditioning
        else reference
    )
    return PreparedContext(context, target, reference)


def pad_context_inputs(tokenizer, inputs: list[ContextInput], device):
    """Pad already encoded canonical inputs; return each input's absolute Q indices."""
    if tokenizer.padding_side != "left":
        raise ValueError("Context collection requires left padding")
    batch = tokenizer.pad(
        [
            {"input_ids": item.input_ids, "attention_mask": [1] * len(item.input_ids)}
            for item in inputs
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)
    width = batch["input_ids"].shape[1]
    positions = [
        [width - len(item.input_ids) + p for p in item.context_positions]
        for item in inputs
    ]
    return batch, positions
