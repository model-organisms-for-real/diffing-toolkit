# Adapted from https://github.com/adamkarvonen/sae_introspect/blob/main/paper_demo/em_demo.py
from diffing.methods.diffing_method import DiffingMethod
from diffing.utils.configs import DictConfig
from pathlib import Path
from typing import Dict
import json

from dataclasses import asdict
from loguru import logger
from omegaconf import OmegaConf

from diffing.utils.agents import DiffingMethodAgent
from diffing.utils.model import load_model_from_config
from .verbalizer import (
    VerbalizerEvalConfig,
    VerbalizerInputInfo,
    run_verbalizer,
    sanitize_lora_name,
)
from .agent import ActivationOracleAgent
from .prompt_context import SCHEMA_VERSION, normalize_conditioning


def parse_prompts(raw_prompts, prefix: str = "") -> list[tuple[str, dict | str | None]]:
    """Return list of (text, tag) tuples. Supports plain strings or {text, tag} dicts.

    Tags may arrive as plain Python dicts (JSON-loaded prompt files) or as
    OmegaConf config nodes (legacy inline YAML); both are normalised to plain
    dicts so the downstream serialisation path stays uniform.
    """
    parsed = []
    for p in raw_prompts:
        if isinstance(p, str):
            parsed.append((prefix + p, None))
        else:
            tag = p.get("tag")
            if OmegaConf.is_config(tag):
                tag = OmegaConf.to_container(tag, resolve=True)
            parsed.append((prefix + p["text"], tag))
    return parsed


# Repo root = parent of diffing-toolkit/. Used to resolve prompt file paths from YAML.
_REPO_ROOT = Path(__file__).resolve().parents[5]


def load_prompts_from_file(rel_path: str) -> list[dict]:
    """Load a JSON list of {text, tag} prompt dicts. Path is repo-root relative unless absolute."""
    path = Path(rel_path)
    if not path.is_absolute():
        path = _REPO_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"Prompt file must be a non-empty JSON list: {path}")
    return data


class ActivationOracleMethod(DiffingMethod):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.results_dir = Path(cfg.diffing.results_dir) / "activation_oracle"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def visualize(self):
        pass

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available results for this method.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        return {}

    def get_agent(self) -> DiffingMethodAgent:
        """Get the agent for the method."""
        return ActivationOracleAgent(cfg=self.cfg)

    # Throughput-only knobs that do not change the numerical results; excluded
    # from the cache hash so bumping them (e.g. for a faster GPU) doesn't force
    # a recompute of existing runs.
    _HASH_EXCLUDED_VERBALIZER_KEYS = ("eval_batch_size",)

    def extra_agent_relevant_cfg(self) -> dict:
        """Include verbalizer config in the hash since it affects agent results."""
        verb_eval = OmegaConf.to_container(self.method_cfg.verbalizer_eval, resolve=True)
        for k in self._HASH_EXCLUDED_VERBALIZER_KEYS:
            verb_eval.pop(k, None)
        relevant = {
            "verbalizer_eval": verb_eval,
            "context_prompts": load_prompts_from_file(self.method_cfg.context_prompts_file),
            "verbalizer_prompts": load_prompts_from_file(self.method_cfg.verbalizer_prompts_file),
        }
        if self.finetuned_model_cfg.activation_scope == "context_content":
            relevant.update({
                "context_schema_version": SCHEMA_VERSION,
                "target": asdict(self.finetuned_model_cfg),
                "reference": asdict(self.base_model_cfg),
                "oracle": self._get_verbalizer_lora_path(),
                "oracle_revision": self.method_cfg.get("oracle_revision"),
                "verbalizer_prefix": self.method_cfg.prefix,
                "measurement_identity": self.method_cfg.get("measurement_identity"),
            })
        return relevant

    def _results_file(self) -> Path:
        return (
            self.results_dir
            / f"{self._get_verbalizer_lora_path().split('/')[-1].replace('/', '_').replace('.', '_')}{'_' if self.agent_cfg_hash else ''}{self.agent_cfg_hash}.json"
        )

    def _load_results(self) -> Dict[str, Dict[str, str]]:
        assert (
            self._results_file().exists()
        ), f"Results file does not exist: {self._results_file()}"
        with self._results_file().open("r") as f:
            return json.load(f)

    def _get_verbalizer_lora_path(self) -> str:
        path = getattr(self.method_cfg.verbalizer_models, self.base_model_cfg.name)
        assert (
            path is not None and path != ""
        ), f"Verbalizer model for {self.base_model_cfg.name} not found"
        return path

    def run(self):
        is_lora = self.finetuned_model_cfg.is_lora
        context_mode = self.finetuned_model_cfg.activation_scope == "context_content"
        if self.finetuned_model_cfg.target_kind == "prompted" and not context_mode:
            raise ValueError("Prompted targets require context_content extraction")
        context_model = None
        if context_mode:
            target = self.finetuned_model_cfg
            if target.target_kind not in {"prompted", "unprompted"} or is_lora:
                raise ValueError("Context mode currently requires a frozen prompted/unprompted target")
            if (target.model_id, target.revision) != (self.base_model_cfg.model_id, self.base_model_cfg.revision):
                raise ValueError("Prompted target and unprompted reference must use identical pinned weights")
            target.conditioning = normalize_conditioning(target.conditioning)
            if (target.target_kind == "prompted") != bool(target.conditioning):
                raise ValueError("Conditioning must be present only for prompted targets")
            import re
            for label, revision in (("weights", target.revision),
                                    ("tokenizer", target.tokenizer_revision),
                                    ("oracle", self.method_cfg.get("oracle_revision"))):
                if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                    raise ValueError(f"Context mode requires a full {label} revision SHA")

        # Layers for activation collection and injection
        model_name = self.base_model_cfg.model_id

        # Skip if results exist and overwrite is disabled
        results_path = self._results_file()
        if results_path.exists() and (not bool(self.method_cfg.overwrite)):
            logger.info(
                f"Results already exist at {results_path}; overwrite=false; skipping run."
            )
            return

        if context_mode:
            # One frozen checkpoint is reused for target, control, and oracle.
            context_model = load_model_from_config(
                self.finetuned_model_cfg,
                extra_adapter_ids=[(self._get_verbalizer_lora_path(), None,
                                    self.method_cfg.oracle_revision)])
            if not context_model.dispatched:
                context_model.dispatch()
            context_model.eval()

        eval_overrides: dict = {}
        if "verbalizer_eval" in self.method_cfg:
            eval_overrides = OmegaConf.to_container(
                self.method_cfg.verbalizer_eval, resolve=True
            )
            assert isinstance(
                eval_overrides, dict
            ), "verbalizer_eval must resolve to a dict"
        if context_mode:
            eval_overrides["activation_scope"] = "context_content"
        config = VerbalizerEvalConfig(
            model_name=model_name,
            num_layers=context_model.num_layers if context_mode else self.base_model.num_layers,
            **eval_overrides,
        )

        # ========================================
        # PROMPT TYPES AND QUESTIONS
        # ========================================

        # IMPORTANT: Context prompts: we send these to the target model and collect activations.
        # Loaded from a file (context_prompts_file); inline lists are not supported.
        context_prompts = parse_prompts(load_prompts_from_file(self.method_cfg.context_prompts_file))
        assert len(context_prompts) > 0, "context_prompts cannot be empty"

        # IMPORTANT: Verbalizer prompts: these are the questions / prompts we send to the verbalizer model, along with context prompt activations
        prefix = self.method_cfg.prefix
        verbalizer_prompts = parse_prompts(
            load_prompts_from_file(self.method_cfg.verbalizer_prompts_file),
            prefix=prefix,
        )

        # Load tokenizer and model(s)
        tokenizer = self.tokenizer
        tokenizer.padding_side = "left"  # run_verbalizer assumes left padding (placeholder positions)
        verbalizer_lora_id = self._get_verbalizer_lora_path()

        if context_mode:
            model = context_model
            verbalizer_lora_name = sanitize_lora_name(verbalizer_lora_id)
            target_lora_name = None
            target_label = self.finetuned_model_cfg.name
            base_model = None
        elif is_lora:
            # LoRA path: load the LoRA's true base (via finetuned_model_cfg.base_model_id,
            # which honors `adapter_base_model_id` overrides) plus the target LoRA (auto-
            # included by load_model_from_config since is_lora) and the verbalizer LoRA.
            # Load the diffing base separately so cross-MO comparisons share one baseline.
            target_lora_id = self.finetuned_model_cfg.model_id
            model = load_model_from_config(
                self.finetuned_model_cfg,
                extra_adapter_ids=[verbalizer_lora_id],
            )
            if not model.dispatched:
                model.dispatch()
            model.eval()

            verbalizer_lora_name = sanitize_lora_name(verbalizer_lora_id)
            target_lora_name = sanitize_lora_name(target_lora_id)
            target_label = target_lora_name

            # Load diffing base separately for "orig" activations.
            base_model = load_model_from_config(self.base_model_cfg)
            if not base_model.dispatched:
                base_model.dispatch()
            base_model.eval()
        else:
            # Full finetune path: load finetuned model with verbalizer adapter,
            # and base model separately for "orig" activations
            logger.info(
                f"Full finetune detected ({self.finetuned_model_cfg.model_id}). "
                "Loading finetuned model with verbalizer adapter and base model separately."
            )
            model = load_model_from_config(
                self.finetuned_model_cfg,
                extra_adapter_ids=[verbalizer_lora_id],
            )
            if not model.dispatched:
                model.dispatch()
            model.eval()

            verbalizer_lora_name = sanitize_lora_name(verbalizer_lora_id)
            target_lora_name = None

            # Load base model for orig activations
            base_model = load_model_from_config(self.base_model_cfg)
            if not base_model.dispatched:
                base_model.dispatch()
            base_model.eval()

            target_label = self.finetuned_model_cfg.name

        logger.info(
            f"Running verbalizer eval for verbalizer: {verbalizer_lora_name}, target: {target_label}"
        )

        # Build context prompts with ground truth
        verbalizer_prompt_infos: list[VerbalizerInputInfo] = []
        for verbalizer_text, verbalizer_tag in verbalizer_prompts:
            for context_text, context_tag in context_prompts:
                formatted_prompt = [
                    {"role": "user", "content": context_text},
                ]
                context_prompt_info = VerbalizerInputInfo(
                    context_prompt=formatted_prompt,
                    ground_truth=target_label,
                    verbalizer_prompt=verbalizer_text,
                    context_prompt_tag=context_tag,
                    verbalizer_prompt_tag=verbalizer_tag,
                )
                verbalizer_prompt_infos.append(context_prompt_info)

        if context_mode:
            from .context_verbalizer import run_context_verbalizer

            results = run_context_verbalizer(
                model, tokenizer, verbalizer_prompt_infos, verbalizer_lora_name,
                config, model.device, conditioning=self.finetuned_model_cfg.conditioning,
                target_kind=self.finetuned_model_cfg.target_kind,
                measurement_identity=self.method_cfg.get("measurement_identity"))
        else:
            results = run_verbalizer(
                model=model,
                tokenizer=tokenizer,
                verbalizer_prompt_infos=verbalizer_prompt_infos,
                verbalizer_lora_path=verbalizer_lora_name,
                target_lora_path=target_lora_name,
                config=config,
                device=model.device,
                is_full_finetune=not is_lora,
                base_model=base_model,
            )

        # Optionally save to JSON

        final_verbalizer_results = {
            "config": asdict(config),
            "results": [asdict(r) for r in results],
        }
        if context_mode:
            final_verbalizer_results["provenance"] = self.extra_agent_relevant_cfg()
        with self._results_file().open("w") as f:
            json.dump(final_verbalizer_results, f, indent=2)
