# Patent SAO Link Prediction

Code structure:

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

## Reproducibility and limitations

UMAP, HDBSCAN, transformer inference, and GPU kernels may still produce small environment-dependent differences. A small public sample may not contain enough unique terms for HDBSCAN; for meaningful execution, include enough patents and terms across all required years.

Stanford CoreNLP and pretrained Hugging Face models are downloaded separately and remain subject to their respective licenses. Check model and data licenses before redistribution.


### Citation
If you find the codes useful, please cite our paper:

```
title = "Discovering technological recombination opportunities using an SAO knowledge graph",
authors = Jeamin Chung, Youngjin Seol, Janghyeok Yoon,
journal = "",
year = ,
doi = ""
```
