# Missing Baseline, Pipeline & Head-Count Control Notebooks — Run Order

These notebooks extend the `baseline_tests/` suite so every missing comparison raised during peer review can be evaluated under the exact frozen splits (`splits_runA.csv`, `splits_runB.csv`), seeds (`42, 43, 44`), and hyperparameters of Experiment 6.

## Recommended Execution Order (GPU -> CPU)

1. **`05a_single_task_sentiment_plain_and_mlm.ipynb`** *(GPU — Run First)*
   - Evaluates **plain** unadapted single-task sentiment (`noMLM + mean` pooling -> `single_task_sent_plain`) and **domain-adapted** single-task sentiment (`MLM + 4-head attn` -> `single_task_sent_mlm_attn`) across `runA` and `runB` (3 seeds each).
   - Trains and caches Domain MLM encoders in `mlm_encoders/mlm_{run}_seed{seed}` (if not already present) and exports Stage-1 validation + test probability caches (`preds/stage1_cache_single_task_sent_*.npz`) so `05c` never needs to retrain Stage 1.
   - Outputs: `results/05a_single_task_sentiment_*.json`, `preds/single_task_sent_*.csv`, `preds/stage1_cache_single_task_sent_*.npz`, `05a_single_task_sentiment_plain_and_mlm_outputs.zip`.

2. **`05b_single_task_intent.ipynb`** *(GPU — Run Second)*
   - Evaluates 6-class **single-task intent** (`single_task_intent_plain` with `noMLM+mean` and `single_task_intent_mlm_attn` with cached `MLM` + single-query 4-head attn) on both the 300-row human gold benchmark and silver-proxy labels.
   - Includes `evaluate_and_save_compat(...)` fallback so it runs cleanly even if an older remote `eval_utils.py` is cloned on Colab.
   - Outputs: `results/05b_single_task_intent_*.json`, `results/cm_intent_05b_*.csv/.png`, `preds/single_task_intent_*.csv`, `preds/stage1_cache_single_task_intent_*.npz`, `05b_single_task_intent_outputs.zip`.

3. **`05c_two_model_and_cascaded_pipeline.ipynb`** *(GPU — Run Third; reuses `05a` & `05b` caches)*
   - **Part A (`05c_pipeline_2model_plain`, `05c_pipeline_2model_mlm_attn`):** Pairs independent `05a` + `05b` predictions into full 2-model non-joint pipelines (no Stage-1 retraining needed if `05a`/`05b` caches are in `preds/`).
   - **Part B (`05c_cascade_sent_to_intent`):** Sequential cascade `Sentiment -> Intent` using prefix conditioning (`[Sentiment: <label>] <text>`), trained with 20% label noise (`COND_NOISE_P = 0.20`) and evaluated strictly on Stage-1 predicted validation/test sentiment (plus an explicit oracle diagnostic check and rhetorical-`?` complaint shift count).
   - **Part C (`05c_cascade_intent_to_sent`):** Sequential cascade `Intent -> Sentiment` (`[Intent: <label>] <text>`), testing whether noisy predicted intent hurts human sentiment accuracy.
   - **Part D:** Exact parameter count (`1x` joint encoder vs `2x` two-model/cascade encoders), batched inference latency comparison, and counterfactual `[Sentiment: Neutral]` vs `[Sentiment: Negative]` qualitative probe.
   - Outputs: `results/05c_*.json`, `preds/pipeline_2model_*.csv`, `preds/cascade_*.csv`, `05c_two_model_and_cascaded_pipeline_outputs.zip`.

4. **`05d_attention_head_count_ablation.ipynb`** *(GPU — Can run anytime after MLM encoders are cached)*
   - Sweeps `HEAD_GRID = [1, 2, 4, 8, 12]` (`head_dim = 768 // h`) for Joint `MLM+attn` using cached `mlm_encoders/` across `runA` and `runB`.
   - Set `SEEDS_TO_EXEC = [42]` in Cell 2 if GPU quota is limited; keep `SEEDS_TO_EXEC = SEEDS` (`[42, 43, 44]`) for full 3-seed means/stds.
   - Outputs: `results/05d_head_ablation_h*.json`, `results/05d_head_ablation_summary.csv`, `results/05d_head_ablation_plot.png`, `preds/MLM+attn_h*.csv`, `05d_attention_head_count_ablation_outputs.zip`.

5. **`06_bootstrap_interaction.ipynb`** *(CPU — Run After Per-Row Prediction CSVs Are Collected in `preds/`)*
   - Computes both paired **row-level** ($B=1{,}000$) and **hierarchical two-level** (paired seed + row resampling) 95% CIs, Holm-Bonferroni & Bonferroni corrections for the $2 \times 2$ factorial ablation (`results/bootstrap_contrasts_{run}.csv`), plus extended paired contrasts comparing Joint `MLM+attn` against `05a`, `05b`, and `05c` (`results/bootstrap_extended_baselines_{run}.csv`).

6. **`07_results_table.ipynb`** *(CPU — Run Last)*
   - Aggregates all `results/*.json` outputs (18 model rows per regime plus `05d` head-count sweep when present) into `results_table.csv`, `results_table_runA.csv`, `results_table_runB.csv`, clean raw-string `results_table.tex`, and `results_head_ablation.csv/.tex`.
   - Automatically prefers individual per-seed JSONs over `*_agg.json` files when both exist for the same notebook tag.

## Shared Cache Locations
- **Domain MLM Checkpoints:** `mlm_encoders/mlm_{run}_seed{seed}` (falls back to `mlm_encoders/mlm_seed{seed}`).
- **Stage-1 Val/Test Caches:** `preds/stage1_cache_single_task_{task}_{variant}_{run}_{seed}.npz` (required for `05c` cache hits).
- **Ablation `.npz` Legacy Caches:** `preds_runA/` and `preds_runB/` (auto-converted to CSV/JSON by `06`).
