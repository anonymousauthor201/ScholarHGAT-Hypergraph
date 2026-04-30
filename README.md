# ScholarHGAT: Heterogeneous Hypergraph Attention Network for Scholar Recommendation and Team Formation

> A graph-based scholar recommendation system built on heterogeneous hypergraph attention networks with TransE-based scoring, supporting both individual scholar recommendation and query-driven team formation.

---

## Overview

**ScholarHGAT** models the academic collaboration landscape as a heterogeneous hypergraph, capturing rich relationships between scholars, NSF awards, and publications. It combines:

- **Hypergraph Attention Network (HGAT)**: learns structural embeddings over multi-way scholar–award–publication hyperedges.
- **TransE-based Scoring**: models directional collaboration affinity in a translational embedding space.
- **QD-DASTF** *(Query-Decomposed Type-Aware Scholar Team Formation)*: a two-stage heuristic algorithm for forming diverse, query-relevant research teams.

The system is evaluated on a private NSF award/publication dataset (~5,099 scholars, 5,631 awards) and the public DBLP Citation Network V15.1.

---

## Repository Structure

```
.
├── to_training_data.py              # Preprocesses raw NSF/DBLP data into training-ready format
├── build_recommendation_inputs.py   # Constructs hypergraph inputs and interaction pairs for model training
├── scholar_hgat.py                  # Core ScholarHGAT model: HGAT encoder + TransE scoring + training loop
├── nsf_team_eval.py                 # Team formation evaluation using QD-DASTF on NSF dataset
└── README.md
```

---

## Pipeline

```
Raw Data (NSF Awards / DBLP)
        │
        ▼
  to_training_data.py          ← Parse and clean raw JSON; build scholar/award/pub mappings
        │
        ▼
build_recommendation_inputs.py ← Construct hyperedges, interaction pairs (train/val/test splits)
        │
        ▼
    scholar_hgat.py             ← Train ScholarHGAT; generate 128-dim scholar embeddings
        │
        ▼
  nsf_team_eval.py              ← Evaluate team formation quality with QD-DASTF
```

---

## Key Features

| Feature | Description |
|---|---|
| Heterogeneous Hypergraph | Models 3-way scholar–award–publication relationships |
| TransE Scoring | Translational scoring for directional collaboration affinity |
| Cold-Start Robustness | Significantly outperforms CF-based baselines (LightGCN, MultVAE) on cold-start scholars |
| Team Formation | QD-DASTF decomposes queries by expertise type for diverse team assembly |
| Full-Ranking Evaluation | Evaluated with full-ranking protocol (not sampled) for rigorous comparison |

---

## Datasets

| Dataset | Scholars | Awards/Interactions | Source |
|---|---|---|---|
| NSF (primary) | ~5,099 | ~5,631 awards | Private NSF award/publication data |
| DBLP | — | V15.1 citations | [DBLP Citation Network](https://www.aminer.org/citation) |

> **Note**: NSF raw data is not included in this repository due to data use restrictions.

---

## Requirements

```bash
pip install torch torch-geometric sentence-transformers numpy scipy
```

Key dependencies:
- Python ≥ 3.8
- PyTorch ≥ 1.12
- torch-geometric
- sentence-transformers (SBERT, for serving embeddings)


## Usage

### 1. Preprocess Data
```bash
python to_training_data.py --data_dir ./data/nsf_raw --output_dir ./data/processed
```

### 2. Build Training Inputs
```bash
python build_recommendation_inputs.py --processed_dir ./data/processed --output_dir ./data/inputs
```

### 3. Train ScholarHGAT
```bash
python scholar_hgat.py --data_dir ./data/inputs --checkpoint_dir ./checkpoints
```

### 4. Evaluate Team Formation
```bash
python nsf_team_eval.py --checkpoint ./checkpoints/best_model.pt --data_dir ./data/inputs
```



