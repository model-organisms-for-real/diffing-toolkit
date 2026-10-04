# PROCESS 1: OLMo
cd /workspace/gks/model-organisms-for-real/diffing-toolkit
export MO_REGISTRY=/workspace/gks/model-organisms-for-real/config/model_registry.json
export CUMPROBS_ROOT=${CUMPROBS_ROOT:-/workspace/model-organisms/cumprobs_v4}  # v4 = V3 quirk descriptions (2026-09-24); v2 = V2 wording
export OLMO_ADL=/workspace/model-organisms/diffing_results/olmo2_1B_sft
export OLMO_CUM=$CUMPROBS_ROOT/olmo2_1B_sft

# don't run these concurrently for caching reasons
bash scripts/cumprobs/run_all_cross_relevance.sh       diff       --adl-base $OLMO_ADL   # 18
bash scripts/cumprobs/run_all_cross_relevance.sh       jlens_diff --adl-base $OLMO_ADL   # 18

# need to explicitly include seedreps:
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py --cross-dir $OLMO_CUM --lens logit_lens --noise-floor --families cake_bake cake_bake_seedrep1 cake_bake_seedrep2 italian_food milsub synth_milsub -o $OLMO_CUM/plots
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py --cross-dir $OLMO_CUM --lens jlens      --noise-floor --families cake_bake cake_bake_seedrep1 cake_bake_seedrep2 italian_food milsub synth_milsub -o $OLMO_CUM/plots

# PROCESS 2: Gemma
cd /workspace/gks/model-organisms-for-real/diffing-toolkit
export MO_REGISTRY=/workspace/gks/model-organisms-for-real/config/model_registry.json
export CUMPROBS_ROOT=${CUMPROBS_ROOT:-/workspace/model-organisms/cumprobs_v4}  # v4 = V3 quirk descriptions (2026-09-24); v2 = V2 wording
export GEMMA_ADL=/workspace/model-organisms/diffing_results/gemma3_1B_ancestor
export GEMMA_CUM=$CUMPROBS_ROOT/gemma3_1B_ancestor

bash scripts/cumprobs/run_all_cross_relevance_gemma.sh diff       --adl-base $GEMMA_ADL    # 9
bash scripts/cumprobs/run_all_cross_relevance_gemma.sh jlens_diff --adl-base $GEMMA_ADL    # 9

uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py --cross-dir $GEMMA_CUM --lens logit_lens --noise-floor -o $GEMMA_CUM/plots
uv run python scripts/cumprobs/plot_cumprobs_raffgraph.py --cross-dir $GEMMA_CUM --lens jlens      --noise-floor -o $GEMMA_CUM/plots
