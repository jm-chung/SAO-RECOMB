from __future__ import annotations

import pickle
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

ACTION_STOPWORDS = {
    "can", "may", "could", "might", "become", "remain", "exist", "have", "possess",
    "include", "comprise", "consist", "contain", "encompass", "incorporate", "involve",
    "provide", "offer", "supply", "afford", "give", "form", "constitute", "make", "up",
    "compose", "construct", "use", "utilize", "employ", "apply", "enable", "allow",
    "locate", "position", "dispose", "connect", "couple", "attach", "join",
}


def _clean_triples(triples) -> pd.DataFrame:
    frame = pd.DataFrame(triples, columns=["reg_num", "subject", "action", "object"])
    frame["reg_num"] = frame["reg_num"].astype(str)
    for column in ("subject", "object"):
        frame[column] = frame[column].map(
            lambda value: " ".join(word for word in str(value).split() if word not in {"invention", "embodiment"})
        )
    frame["action"] = frame["action"].map(
        lambda value: " ".join(word for word in str(value).split() if word not in ACTION_STOPWORDS)
    )
    return frame[(frame.subject != "") & (~frame.action.isin(["", "be"])) & (frame.object != "")].copy()


def _cluster_column(
    former: pd.DataFrame,
    latter: pd.DataFrame,
    column: str,
    model: SentenceTransformer,
    prefix_counter: dict[str, int],
    top_k: int,
    encode_batch_size: int,
    n_neighbors: int,
    n_components: int,
    min_dist: float,
    min_cluster_size: int,
    min_samples: int,
    seed: int,
):
    terms = former[column].dropna().unique()
    if len(terms) < max(3, min_cluster_size):
        raise ValueError(f"Not enough unique {column} terms for clustering: {len(terms)}")

    embeddings = model.encode(terms, batch_size=encode_batch_size, normalize_embeddings=True, show_progress_bar=True)
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(terms) - 1),
        n_components=min(n_components, max(2, len(terms) - 2)),
        min_dist=min_dist,
        random_state=seed,
    )
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        core_dist_n_jobs=-1,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)

    numeric_to_id: dict[int, str | None] = {-1: None}
    for numeric_id in sorted(set(labels) - {-1}):
        prefix_counter[column] += 1
        numeric_to_id[numeric_id] = f"{column[0]}{prefix_counter[column]}"

    info = pd.DataFrame({column: terms, "numeric_id": labels, "embedding": list(reduced)})
    frequency = former[["reg_num", column]].drop_duplicates().groupby(column).size().rename("doc_freq").reset_index()
    info = info.merge(frequency, on=column, how="left").fillna({"doc_freq": 0})

    labels_out: dict[str, str] = {}
    for numeric_id, cluster_id in numeric_to_id.items():
        if numeric_id == -1 or cluster_id is None:
            continue
        members = info[info.numeric_id == numeric_id].copy()
        max_frequency = members.doc_freq.max()
        candidates = members[members.doc_freq == max_frequency].copy()
        medoid = clusterer.weighted_cluster_medoid(numeric_id)
        if medoid is not None and len(candidates) > 1:
            distances = cosine_distances(medoid.reshape(1, -1), np.stack(candidates.embedding)).ravel()
            candidates = candidates.iloc[np.argsort(distances)]
        remaining = members[~members[column].isin(candidates[column])].sort_values("doc_freq", ascending=False)
        ordered = pd.concat([candidates, remaining]).drop_duplicates(subset=[column])
        labels_out[cluster_id] = ", ".join(ordered[column].head(top_k).tolist())

    former_map = {term: numeric_to_id.get(int(label)) for term, label in zip(terms, labels)}
    former_ids = former[column].map(former_map)

    new_terms = latter[column].dropna().unique()
    latter_map: dict[str, str | None] = {}
    if len(new_terms):
        new_embeddings = model.encode(new_terms, batch_size=encode_batch_size, normalize_embeddings=True, show_progress_bar=True)
        new_reduced = reducer.transform(new_embeddings)
        new_labels, _ = hdbscan.approximate_predict(clusterer, new_reduced)
        latter_map = {term: numeric_to_id.get(int(label)) for term, label in zip(new_terms, new_labels)}
    latter_ids = latter[column].map(latter_map)
    return former_ids, latter_ids, labels_out


def cluster_triples(input_csv: str | Path, triples_pkl: str | Path, output_pkl: str | Path,
                     labels_pkl: str | Path, settings: dict, seed: int = 42) -> None:
    patents = pd.read_csv(input_csv)
    patents["reg_num"] = patents["reg_num"].astype(str)
    patents["pat_year"] = pd.to_numeric(patents["pat_year"], errors="raise").astype(int)
    with Path(triples_pkl).open("rb") as file:
        triples = pickle.load(file)
    frame = _clean_triples(triples)

    fit_ids = set(patents.loc[patents.pat_year.between(settings["fit_start_year"], settings["fit_end_year"]), "reg_num"])
    predict_ids = set(patents.loc[patents.pat_year.between(settings["predict_start_year"], settings["predict_end_year"]), "reg_num"])
    former = frame[frame.reg_num.isin(fit_ids)].copy()
    latter = frame[frame.reg_num.isin(predict_ids)].copy()

    model = SentenceTransformer(settings["model_name"])
    counters = {"subject": 0, "action": 0, "object": 0}
    cluster_labels: dict[str, str] = {}
    for column in ("subject", "action", "object"):
        former[column], latter[column], labels = _cluster_column(
            former, latter, column, model, counters,
            top_k=settings["label_top_k"], encode_batch_size=settings["encode_batch_size"],
            n_neighbors=settings["n_neighbors"], n_components=settings["n_components"],
            min_dist=settings["min_dist"], min_cluster_size=settings["min_cluster_size"],
            min_samples=settings["min_samples"], seed=seed,
        )
        cluster_labels.update(labels)

    former = former.dropna(subset=["subject", "action", "object"]).drop_duplicates()
    latter = latter.dropna(subset=["subject", "action", "object"]).drop_duplicates()
    former = former[former.subject != former.object]
    latter = latter[latter.subject != latter.object]

    Path(output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_pkl).open("wb") as file:
        pickle.dump([former.values, latter.values], file)
    with Path(labels_pkl).open("wb") as file:
        pickle.dump(cluster_labels, file)
