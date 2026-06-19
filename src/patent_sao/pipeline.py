from __future__ import annotations

import argparse
import logging

from .cluster import cluster_triples
from .config import load_config
from .dataset import generate_dataset
from .extract import extract_triples
from .train import train_and_evaluate

LOGGER = logging.getLogger(__name__)


def run_stage(config: dict, stage: str):
    paths, seed = config["paths"], config["project"]["seed"]
    if stage in {"extract", "all"}:
        extract_triples(paths["input_csv"], paths["triples_pkl"], **config["extraction"])
    if stage in {"cluster", "all"}:
        cluster_triples(paths["input_csv"], paths["triples_pkl"], paths["clustered_triples_pkl"],
                         paths["cluster_labels_pkl"], config["clustering"], seed)
    if stage in {"dataset", "all"}:
        counts = generate_dataset(paths["input_csv"], paths["clustered_triples_pkl"], paths["dataset_pkl"],
                                  seed=seed, **config["dataset"])
        LOGGER.info("Dataset class counts: %s", counts)
    if stage in {"train", "all"}:
        metrics = train_and_evaluate(paths["dataset_pkl"], paths["cluster_labels_pkl"], paths["model_path"],
                                     paths["metrics_json"], paths["predictions_csv"], paths["pr_curve_png"],
                                     config["training"], seed)
        LOGGER.info("Test metrics: %s", metrics)


def main():
    parser = argparse.ArgumentParser(description="Patent SAO link-prediction pipeline")
    parser.add_argument("stage", choices=["extract", "cluster", "dataset", "train", "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_stage(load_config(args.config), args.stage)


if __name__ == "__main__":
    main()
