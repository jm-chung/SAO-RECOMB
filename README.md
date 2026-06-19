# Patent SAO Link Prediction

A reproducible Python pipeline refactored from exploratory notebooks. It starts with a public `total_pat.csv` file and runs:

1. SAO (Subject–Action–Object) extraction with Stanford CoreNLP OpenIE
2. PatentSBERTa embedding, UMAP reduction, and HDBSCAN clustering
3. Temporal positive/negative triple dataset generation
4. Semantic Scorer + Structural Scorer ensemble training and test evaluation

No database connection, credential, private host, or notebook-only inspection code is included.

## Input

Put the sample file at `data/raw/total_pat.csv` with these columns:

```text
reg_num,pat_year,patent_title,patent_abstract
```

The default configuration expects data covering at least 2010–2021.

## Installation

Python 3.10 or 3.11 is recommended. SAO extraction also requires Java 11 or later.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
python -m stanza.install_corenlp
```

For GPU training, install the PyTorch build matching your CUDA environment before installing the remaining requirements.

## Run the full pipeline

Run from the repository root:

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"
python scripts/run_pipeline.py all --config configs/default.yaml

# Linux/macOS
PYTHONPATH=src python scripts/run_pipeline.py all --config configs/default.yaml
```

Individual stages can be rerun without repeating earlier stages:

```bash
PYTHONPATH=src python scripts/run_pipeline.py extract
PYTHONPATH=src python scripts/run_pipeline.py cluster
PYTHONPATH=src python scripts/run_pipeline.py dataset
PYTHONPATH=src python scripts/run_pipeline.py train
```

## Main outputs

- `data/interim/triples.pkl`: extracted SAO triples
- `data/processed/clustered_triples.pkl`: cluster-ID triples for former/latter periods
- `data/processed/cluster_labels_top3.pkl`: human-readable labels for cluster IDs
- `data/processed/dataset_1_3.pkl`: training examples with an approximately 1:3 positive/negative ratio
- `models/SAO_BC_best.pt`: best validation checkpoint
- `outputs/test_metrics.json`: test metrics
- `outputs/test_predictions.csv`: labels, probabilities, and predictions
- `outputs/pr_curve.png`: precision-recall curve

## Configuration notes

All paths and parameters are in `configs/default.yaml`. The original notebooks used very large epoch counts and a 32 GB CoreNLP heap. Public defaults are reduced so that the pipeline is easier to test; increase them for a full experiment.

The temporal dataset logic is preserved: each period uses four input years and the following two output years, shifted by one year across five periods. Triples persisting or newly appearing in the output period are positive. Disappearing triples, self-linked triples, and sampled unobserved combinations are negative.

## Reproducibility and limitations

UMAP, HDBSCAN, transformer inference, and GPU kernels may still produce small environment-dependent differences. A small public sample may not contain enough unique terms for HDBSCAN; for meaningful execution, include enough patents and terms across all required years.

Stanford CoreNLP and pretrained Hugging Face models are downloaded separately and remain subject to their respective licenses. Check model and data licenses before redistribution.
