from __future__ import annotations

import pickle
import random
from collections import Counter
from pathlib import Path

import pandas as pd


def _sample_random_negatives(existing: set[str], subjects: list[str], action_objects: list[str], count: int,
                             rng: random.Random) -> list[list[str]]:
    if count <= 0:
        return []
    maximum = len(subjects) * len(action_objects) - len(existing)
    if count > maximum:
        raise ValueError(f"Requested {count} random negatives, but at most {maximum} are available.")
    sampled: set[str] = set()
    attempts = 0
    max_attempts = max(10000, count * 100)
    while len(sampled) < count and attempts < max_attempts:
        sequence = f"{rng.choice(subjects)};{rng.choice(action_objects)}"
        if sequence not in existing:
            sampled.add(sequence)
        attempts += 1
    if len(sampled) < count:
        for subject in subjects:
            for action_object in action_objects:
                sequence = f"{subject};{action_object}"
                if sequence not in existing:
                    sampled.add(sequence)
                    if len(sampled) == count:
                        break
            if len(sampled) == count:
                break
    return [sequence.split(";") for sequence in sampled]


def generate_dataset(input_csv: str | Path, clustered_pkl: str | Path, output_pkl: str | Path,
                     periods: int = 5, input_year_count: int = 4, output_year_count: int = 2,
                     negative_ratio: int = 3, seed: int = 42) -> Counter:
    patents = pd.read_csv(input_csv)
    patents["reg_num"] = patents["reg_num"].astype(str)
    patents["pat_year"] = pd.to_numeric(patents["pat_year"], errors="raise").astype(int)
    with Path(clustered_pkl).open("rb") as file:
        former, _ = pickle.load(file)
    triples = pd.DataFrame(former, columns=["reg_num", "subject", "action", "object"])
    triples["reg_num"] = triples["reg_num"].astype(str)

    rows = []
    base_year = 2010
    for period in range(periods):
        input_years = range(base_year + period, base_year + period + input_year_count)
        output_start = base_year + period + input_year_count
        output_years = range(output_start, output_start + output_year_count)
        input_ids = set(patents.loc[patents.pat_year.isin(input_years), "reg_num"])
        output_ids = set(patents.loc[patents.pat_year.isin(output_years), "reg_num"])
        input_sequences = set(triples[triples.reg_num.isin(input_ids)].iloc[:, 1:].astype(str).agg(";".join, axis=1))
        output_sequences = set(triples[triples.reg_num.isin(output_ids)].iloc[:, 1:].astype(str).agg(";".join, axis=1))

        for sequence in input_sequences:
            head, relation, tail = sequence.split(";")
            rows.append((f"p{period + 1}", sequence, int(sequence in output_sequences), 1 if sequence in output_sequences else 3))
            rows.extend([(f"p{period + 1}", f"{head};{relation};{head}", 0, 4),
                         (f"p{period + 1}", f"{tail};{relation};{tail}", 0, 4)])
        for sequence in output_sequences - input_sequences:
            head, relation, tail = sequence.split(";")
            rows.append((f"p{period + 1}", sequence, 1, 2))
            rows.extend([(f"p{period + 1}", f"{head};{relation};{head}", 0, 4),
                         (f"p{period + 1}", f"{tail};{relation};{tail}", 0, 4)])

    labelled = pd.DataFrame(rows, columns=["period", "sequence", "label", "type"]).drop_duplicates()
    unique = labelled.sort_values(["label", "period", "sequence"], ascending=False).drop_duplicates("sequence")
    X = [sequence.split(";") for sequence in unique.sequence]
    y = unique.label.astype(int).tolist()

    counts = Counter(y)
    target_negatives = counts[1] * negative_ratio
    required = target_negatives - counts[0]
    if required > 0:
        non_self = unique[unique.type != 4]
        subjects = sorted({sequence.split(";")[0] for sequence in unique.sequence if sequence.split(";")[0][0] != sequence.split(";")[2][0]})
        action_objects = sorted({";".join(sequence.split(";")[1:]) for sequence in unique.sequence if sequence.split(";")[0][0] != sequence.split(";")[2][0]})
        existing = set(non_self.sequence)
        extra = _sample_random_negatives(existing, subjects, action_objects, required, random.Random(seed))
        X.extend(extra)
        y.extend([0] * len(extra))

    output = Path(output_pkl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump([X, y], file)
    return Counter(y)
