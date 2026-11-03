# Multi-Task Pooling for Joint Sentiment and Intent Analysis in Code-Mixed Sinhala

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗%20Transformers-XLM--RoBERTa-yellow.svg)](https://huggingface.co/FacebookAI/xlm-roberta-base)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Data License: CC BY-NC 4.0](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> **Final-Year Undergraduate Research Project (`CSC406S6` / `CSC406M6`)**  
> **Author:** S. T. Chanditha (`2021/CSC/053`)  
> **Supervisor:** Dr. K. Sarveswaran  
> **Institution:** Department of Computer Science, University of Jaffna, Sri Lanka  

---

## Overview

Automated customer care and social media triaging in Sri Lanka require natural language understanding (NLU) models capable of simultaneously identifying a user's **emotional tone (sentiment)** and **communicative goal (intent)**. However, informal online Sinhala text is heavily code-mixed—blending native Sinhala abugida script (`U+0D80`–`U+0DFF`), Romanized Sinhala ("Singlish"), English technical terms, conversational punctuation, and emojis—and lacks human-annotated multi-task benchmarks.

This repository provides a complete multi-task learning framework for joint **4-class sentiment** (*Neutral, Negative, Positive, Conflict*) and **6-class communicative intent** (*Inquiry, Complaint, Feedback, Suggestion, Greeting, Other*) classification on Sri Lankan telecommunications customer care comments.

### Key Highlights
* **Script-Preserving Normalization:** Preserves Zero-Width Joiners (`U+200D` / `U+200C`) required for valid Sinhala consonant conjuncts (*yansaya, rakaransaya, bandhana akuru*), alongside experiments comparing alphanumeric-only vs. punctuation/emoji-retaining normalization.
* **Hybrid Intent Bootstrapping & Failure Analysis:** Documents why cascading Singlish transliteration (`IndoNLP BERT`) $\to$ translation (`NLLB-200`) $\to$ small-LLM prompting (`Qwen2-1.5B`) fails on informal Singlish, and implements a direct zero-shot cross-lingual NLI (`joeddav/xlm-roberta-large-xnli`) + domain regex override pipeline with confidence filtering ($p_{\max} \ge 0.50$).
* **300-Row Human-Verified Benchmark:** Pairs human-verified sentiment labels from the SOLD Customer Care Social Media (CCMS) corpus with native-speaker manual intent annotations on a strictly held-out 300-row test set.
* **Domain-Adaptive MLM & Task-Specific Attention Pooling:** Evaluates a two-stage training paradigm combining in-domain Masked Language Modeling (MLM, 15% dynamic masking) with a 4-head task-specific attention pooler using independent learnable query vectors ($\mathbf{q}_{\text{sent}}, \mathbf{q}_{\text{int}}$).

---

## Repository Structure

```text
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── processed/
│   │   └── sinhala_silver_labels.csv                 # 5,881 confidence-filtered multi-task instances
│   └── benchmark/
│       └── annotation_with_sentiment_intent.csv      # 300-row human-verified gold test benchmark
│
└── notebooks/
    ├── 01_data_curation/
    │   ├── 01_sold_ccms_eda_and_split.ipynb
    │   └── 02_singlish_transliteration_nllb_llm_pipeline.ipynb
    │
    ├── 02_baseline_models/
    │   ├── 03_cartesian_12class_single_head.ipynb
    │   ├── 04_cross_lingual_shared_trunk_transfer.ipynb
    │   └── 05_asymmetric_dual_head_holdout.ipynb
    │
    ├── 03_cross_validation_resampling/
    │   ├── 06_asymmetric_dual_head_2fold_cv_oversampling.ipynb
    │   └── 07_asymmetric_dual_head_5fold_cv_oversampling.ipynb
    │
    └── 04_domain_mlm_and_attention_ablation/
        ├── 08_domain_mlm_single_query_attention_prototype.ipynb
        ├── 09_factorial_ablation_mlm_x_pooling_clean_zwj.ipynb
        └── 10_factorial_ablation_mlm_x_pooling_emoji_punct.ipynb
```

---

## Dataset & Taxonomies

Starting from the **13,518-sentence Customer Care Social Media (CCMS)** subset of the [SOLD benchmark (Ranasinghe et al., 2024)](https://doi.org/10.1007/s10579-024-09723-1), confidence filtering ($p_{\max} \ge 0.50$) retains **5,881 multi-task instances**. Before any domain MLM pretraining or fine-tuning, **300 rows** were isolated and manually annotated for communicative intent by a native Sinhala speaker.

| Task | Classes | 300-Row Human Gold Test Distribution |
| :--- | :--- | :--- |
| **Sentiment (4 classes)**<br/>*(Human CCMS Ground Truth)* | `Neutral`, `Negative`, `Positive`, `Conflict` | **Neutral:** 184 (61.3%) · **Negative:** 90 (30.0%) · **Positive:** 25 (8.3%) · **Conflict:** 1 (0.3%) |
| **Intent (6 classes)**<br/>*(Human Gold Annotated)* | `Feedback`, `Inquiry`, `Complaint`, `Suggestion`, `Greeting`, `Other` | **Feedback:** 94 (31.3%) · **Inquiry:** 78 (26.0%) · **Complaint:** 61 (20.3%) · **Suggestion:** 31 (10.3%) · **Greeting:** 23 (7.7%) · **Other:** 13 (4.3%) |

* **Silver vs. Gold Agreement on the 300 Test Rows:**
  * **Sentiment:** 100.0% agreement ($\kappa = 1.000$)
  * **Intent:** 45.0% raw agreement ($\kappa = 0.293$) — zero-shot NLI + question-mark regex rules over-predicted `Inquiry` (55.7% silver vs. 26.0% human gold) and under-predicted indirect `Complaint` and `Feedback`.

---

## Experimental Results

### 1. Progression of Multi-Task Architectures (Internal Silver Splits, $N = 5,881$)

| Notebook & Architecture | Evaluation Protocol | Joint Acc (%) | Sent Acc (%) | Intent Acc (%) | Sent Macro-F1 | Intent Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `03` Cartesian 12-Class Single-Head *(1,000-row pilot)* | 85/15 Val ($N=149$) | 52.35 | — | — | 0.1200 *(12-cls)* | — |
| `04` Cross-Lingual Shared-Trunk (`GoEmotions` + `CLINC150`) | 70/15/15 Test ($N=883$) | 57.64 | 74.97 | 73.73 | 0.4822 | 0.5180 |
| `05` Asymmetric Dual-Head (`[CLS]`, Label Smoothing) | 70/15/15 Test ($N=883$) | 63.19 | 75.42 | 80.97 | 0.4479 | 0.6799 |
| `06` Asymmetric Dual-Head + Composite `RandomOverSampler` | **2-Fold Stratified CV** | 61.86 ± 0.23 | 75.36 ± 0.40 | 81.13 ± 0.51 | **0.5337 ± 0.0012** | **0.7114 ± 0.0050** |
| `07` Asymmetric Dual-Head + Composite `RandomOverSampler` | **5-Fold CV (Fold 1)** | **64.83** | **77.15** | **83.69** | **0.5435** | **0.7292** |

---

### 2. Multi-Seed $2 \times 2$ Factorial Ablation on the 300-Row Human Benchmark

We evaluate four variants crossing **Domain Pretraining (`MLM` vs. `noMLM`)** and **Pooling (`attn` vs. `mean`)** across **3 random seeds (`42, 43, 44`; 12 runs per regime)**:
* **Run A (`09_factorial_ablation...clean_zwj.ipynb`):** Alphanumeric + Sinhala + ZWJ normalization ($N = 5,723$: 4,609 train / 814 val / 300 gold test).
* **Run B (`10_factorial_ablation...emoji_punct.ipynb`):** Preserves conversational punctuation (`?!.`) and Unicode emojis + gold-safe deduplication ($N = 5,834$: 4,703 train / 831 val / 300 gold test).

| Preprocessing Regime | Variant | Gold Sent Acc | Gold Sent Macro-F1 (4-cls) | Gold Sent Macro-F1 (3-cls)* | Gold Intent Acc | Gold Intent Macro-F1 | Gold Joint Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run A**<br/>*(Clean ZWJ, $N=5,723$)* | **`MLM+attn` (Ours)** | **0.776 ± 0.013** | **0.539 ± 0.017** | **0.719** | 0.454 ± 0.006 | 0.398 ± 0.009 | **0.401 ± 0.004** |
| | `noMLM+attn` | 0.713 ± 0.005 | 0.484 ± 0.005 | 0.645 | 0.461 ± 0.004 | 0.403 ± 0.007 | 0.383 ± 0.005 |
| | `MLM+mean` | 0.740 ± 0.020 | 0.501 ± 0.017 | 0.668 | **0.463 ± 0.010** | **0.408 ± 0.008** | 0.392 ± 0.006 |
| | `noMLM+mean` | 0.713 ± 0.005 | 0.487 ± 0.004 | 0.649 | 0.454 ± 0.002 | 0.397 ± 0.007 | 0.374 ± 0.003 |
| **Run B**<br/>*(Emoji + Punct Kept, $N=5,834$)* | **`MLM+attn` (Ours)** | **0.751 ± 0.002** | **0.512 ± 0.005** | **0.683** | 0.474 ± 0.004 | 0.410 ± 0.010 | 0.401 ± 0.007 |
| | `noMLM+attn` | 0.722 ± 0.007 | 0.488 ± 0.003 | 0.651 | 0.476 ± 0.017 | 0.412 ± 0.027 | 0.394 ± 0.016 |
| | `MLM+mean` | 0.738 ± 0.008 | 0.500 ± 0.007 | 0.667 | **0.484 ± 0.002** | **0.429 ± 0.006** | **0.410 ± 0.003** |
| | `noMLM+mean` | 0.728 ± 0.007 | 0.494 ± 0.006 | 0.659 | 0.469 ± 0.006 | 0.407 ± 0.007 | 0.390 ± 0.003 |

*\*Note: Because the 300-row test split contains only $N=1$ instance of the rare `Conflict` class (where all models score 0.000 recall), 3-class Sentiment Macro-F1 across populated classes (`Neutral, Negative, Positive`) is reported alongside 4-class Macro-F1. On Seed 42 (Run A), `MLM+attn` reaches **79.3% sentiment accuracy**, **0.562 4-class F1**, and **0.749 3-class Macro-F1**.*

#### Statistical Significance (Paired Row-Level Bootstrap, $B = 1{,}000$)
* **`MLM+attn` vs. `noMLM+attn` (Sentiment):**
  * **Run A:** $\Delta\text{Macro-F1} = \mathbf{+0.0545}$, 95% CI **$[+0.035, +0.100]$** *(statistically significant, excludes 0)*.
  * **Run B:** $\Delta\text{Macro-F1} = \mathbf{+0.0237}$, 95% CI **$[+0.004, +0.057]$** *(statistically significant, excludes 0)*.
* **Key Takeaway:** Applying task-specific attention pooling to an unadapted encoder (`noMLM+attn`) yields no gain over mean pooling (`noMLM+mean`), whereas pretraining the encoder with in-domain MLM (`MLM+attn`) unlocks a consistent, statistically significant sentiment improvement. Meanwhile, gold intent performance is bounded by the ~45% agreement ceiling of the zero-shot silver training labels.

---

## Getting Started

### 1. Installation
```bash
git clone https://github.com/ThiwankaChanditha/Sinhala-joint-sentiment-and-intent.git
cd Sinhala-joint-sentiment-and-intent
pip install -r requirements.txt
```

### 2. Running the Factorial Ablation Study
Open `notebooks/04_domain_mlm_and_attention_ablation/09_factorial_ablation_mlm_x_pooling_clean_zwj.ipynb` (or `10_...emoji_punct.ipynb`) in Jupyter, Google Colab, or Kaggle (T4 GPU recommended) and point `CSV_PATH` and `GOLD_CSV` to:
```python
CSV_PATH = "data/processed/sinhala_silver_labels.csv"
GOLD_CSV = "data/benchmark/annotation_with_sentiment_intent.csv"
```
All 12 runs (4 variants $\times$ 3 seeds) cache their intermediate MLM encoders and prediction `.npz` arrays automatically so runs can be resumed seamlessly.

---

## Citation

If you use this code, the 300-row human intent benchmark, or our findings in your research, please cite both our work and the original SOLD dataset:

```bibtex
@misc{chanditha2026multitask,
  author       = {Chanditha, S. T. and Sarveswaran, K.},
  title        = {Multi-Task Pooling for Joint Sentiment and Intent Analysis in Sinhala Language},
  year         = {2026},
  institution  = {Department of Computer Science, University of Jaffna}
}

@article{ranasinghe2024sold,
  author  = {Ranasinghe, Tharindu and Anuradha, Isuri and Premasiri, Damith and Silva, Kanishka and Hettiarachchi, Hansi and Uyangodage, Lasitha and Zampieri, Marcos},
  title   = {{SOLD}: Sinhala Offensive Language Dataset},
  journal = {Language Resources and Evaluation},
  year    = {2024},
  doi     = {10.1007/s10579-024-09723-1}
}
```

---

## License

* **Source Code & Notebooks:** Licensed under the [MIT License](LICENSE).
* **Dataset & Annotations:** The derived multi-task intent annotations and the 300-row human evaluation benchmark are released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license for academic research purposes, subject to the original terms of the [SOLD / CCMS dataset (Ranasinghe et al., 2024)](https://doi.org/10.1007/s10579-024-09723-1).
