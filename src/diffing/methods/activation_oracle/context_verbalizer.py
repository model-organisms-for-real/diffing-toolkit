"""Context-only AO for frozen prompted models and their unprompted controls."""

from contextlib import contextmanager

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper

from .prompt_context import ACTIVATION_LABELS, pad_context_inputs, prepare_context
from .utils.activation_utils import collect_activations_multiple_layers
from .utils.eval import run_evaluation
from .verbalizer import VerbalizerResults, create_verbalizer_inputs


@contextmanager
def adapter_state(model, *, disabled=False, adapter=None):
    """Restore every adapter layer's selection and enable state, even on failure."""
    native = getattr(model, "_model", model)
    layers = [
        m
        for m in native.modules()
        if isinstance(m, (BaseTunerLayer, ModulesToSaveWrapper))
    ]
    state = [(m, list(m.active_adapters), m.disable_adapters) for m in layers]
    try:
        if adapter is not None:
            model.set_adapter(adapter)
        for layer in layers:
            layer.enable_adapters(enabled=not disabled)
        yield
    finally:
        for layer, active, was_disabled in state:
            layer.set_adapter(
                active if isinstance(layer, BaseTunerLayer) else active[0]
            )
            layer.enable_adapters(enabled=not was_disabled)


def collect_context_activations(model, tokenizer, inputs, config, device):
    batch, positions = pad_context_inputs(tokenizer, inputs, device)
    submodules = {layer: model.layers[layer]._module for layer in config.act_layers}
    # Separate forwards, no shared past_key_values or generation state.
    batch["use_cache"] = False
    with adapter_state(model, disabled=True):
        return collect_activations_multiple_layers(
            model,
            submodules,
            batch,
            min_offset=None,
            max_offset=None,
            context_positions=positions,
        )


def run_context_verbalizer(
    model,
    tokenizer,
    verbalizer_prompt_infos,
    verbalizer_lora_path,
    config,
    device,
    *,
    conditioning=None,
    target_kind="prompted",
    measurement_identity=None,
):
    if config.activation_scope != "context_content":
        raise ValueError("Context verbalizer requires context_content")
    if config.add_response_to_context_prompt:
        raise ValueError("Generated answers cannot enter context_content")
    if target_kind not in {"prompted", "unprompted"}:
        raise ValueError(
            "Context mode currently supports frozen prompted/unprompted models"
        )
    if (target_kind == "prompted") != bool(conditioning):
        raise ValueError("Only prompted targets must have conditioning")
    native = getattr(model, "_model", model)
    max_length = getattr(native.config, "max_position_embeddings", None)
    prepared = {}
    for info in verbalizer_prompt_infos:
        if len(info.context_prompt) != 1 or info.context_prompt[0]["role"] != "user":
            raise ValueError(
                "Display context must contain exactly one user message with Q"
            )
        q = info.context_prompt[0]["content"]
        if q not in prepared:
            prepared[q] = prepare_context(
                tokenizer,
                q,
                conditioning,
                max_length=max_length,
                add_generation_prompt=config.add_generation_prompt,
                enable_thinking=config.enable_thinking,
            )

    results = []
    for start in range(0, len(verbalizer_prompt_infos), config.eval_batch_size):
        infos = verbalizer_prompt_infos[start : start + config.eval_batch_size]
        contexts = [prepared[info.context_prompt[0]["content"]] for info in infos]
        # V changes the oracle's question, never the target forward. Collect each
        # Q only once within the batch even when it is paired with many questions.
        unique = list({c.display_context: c for c in contexts}.values())
        indices = {c.display_context: b for b, c in enumerate(unique)}
        acts = {}
        if "lora" in config.activation_input_types:
            acts["lora"] = collect_context_activations(
                model, tokenizer, [c.target for c in unique], config, device
            )
        if "orig" in config.activation_input_types:
            acts["orig"] = (
                acts["lora"]
                if conditioning is None and "lora" in acts
                else collect_context_activations(
                    model, tokenizer, [c.reference for c in unique], config, device
                )
            )
        # Difference is optional and blocked per example when Q does not align.
        # Raw conditioned and baseline results remain available in that case.
        if "diff" in config.activation_input_types:
            acts["diff"] = {
                layer: [
                    (
                        acts["lora"][layer][b] - acts["orig"][layer][b]
                        if c.aligned
                        else None
                    )
                    for b, c in enumerate(unique)
                ]
                for layer in config.act_layers
            }

        datapoints, buckets = [], {}
        for b, (info, context) in enumerate(zip(infos, contexts)):
            for act_key, by_layer in acts.items():
                if act_key == "diff" and not context.aligned:
                    continue
                side = context.reference if act_key == "orig" else context.target
                key = (b, act_key)
                selected = {
                    layer: values[indices[context.display_context]].unsqueeze(0)
                    for layer, values in by_layer.items()
                }
                new = create_verbalizer_inputs(
                    selected,
                    side.context_token_ids,
                    info.verbalizer_prompt,
                    config.active_layer,
                    config.active_layer,
                    tokenizer,
                    config,
                    base_meta={"combo_index": b, "act_key": act_key},
                )
                if not new:
                    raise ValueError("Probe settings select no context activations")
                datapoints.extend(new)
                buckets[key] = {
                    "token": [None] * len(side.context_token_ids),
                    "segment": [],
                    "full_seq": [],
                    "probe_counts": {
                        kind: sorted(
                            {
                                len(dp.context_positions)
                                for dp in new
                                if dp.meta_info["dp_kind"] == kind
                            }
                        )
                        for kind in config.verbalizer_input_types
                    },
                }

        with adapter_state(model, adapter=verbalizer_lora_path):
            responses = run_evaluation(
                eval_data=datapoints,
                model=model,
                tokenizer=tokenizer,
                submodule=model.layers[config.injection_layer]._module,
                device=device,
                dtype=native.dtype,
                global_step=-1,
                lora_path=verbalizer_lora_path,
                eval_batch_size=config.eval_batch_size,
                steering_coefficient=config.steering_coefficient,
                generation_kwargs=config.verbalizer_generation_kwargs,
            )
        for response in responses:
            meta = response.meta_info
            bucket = buckets[(meta["combo_index"], meta["act_key"])]
            kind = meta["dp_kind"]
            if kind == "tokens":
                bucket["token"][meta["token_index"]] = response.api_response
            else:
                bucket[kind].append(response.api_response)
        for (b, act_key), bucket in buckets.items():
            info, context = infos[b], contexts[b]
            side = context.reference if act_key == "orig" else context.target
            results.append(
                VerbalizerResults(
                    verbalizer_lora_path=verbalizer_lora_path,
                    target_lora_path=None,
                    context_prompt=info.context_prompt,
                    display_context=context.display_context,
                    act_key=act_key,
                    verbalizer_prompt=info.verbalizer_prompt,
                    ground_truth=info.ground_truth,
                    num_tokens=len(side.context_token_ids),
                    token_responses=bucket["token"],
                    segment_responses=bucket["segment"],
                    full_sequence_responses=bucket["full_seq"],
                    context_input_ids=side.context_token_ids,
                    context_prompt_tag=info.context_prompt_tag,
                    verbalizer_prompt_tag=info.verbalizer_prompt_tag,
                    activation_scope="context_content",
                    target_kind=target_kind,
                    comparison_kind=(
                        "prompt_vs_unprompted" if conditioning else "unprompted_control"
                    ),
                    activation_label=(
                        ACTIVATION_LABELS[act_key]
                        if conditioning
                        else "unconditioned_context"
                    ),
                    measurement_identity=measurement_identity,
                    context_diagnostics={
                        "target": context.target.diagnostics(),
                        "reference": context.reference.diagnostics(),
                        "alignment": "matched" if context.aligned else "blocked",
                        "probe_token_counts": bucket["probe_counts"],
                    },
                )
            )
    return results
