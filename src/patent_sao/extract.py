from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Iterable, Optional

import nltk
import pandas as pd
import stanza
from nltk.corpus import stopwords
from stanza.server import CoreNLPClient
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
BE_VERBS = {"am", "is", "are", "was", "were", "be", "been", "being"}
DOMAIN_STOPWORDS = {"system", "method", "apparatus", "device", "invention", "embodiment"}


def _load_stopwords() -> set[str]:
    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    return (set(words) - BE_VERBS) | DOMAIN_STOPWORDS


def validate_patent_csv(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"reg_num", "pat_year", "patent_title", "patent_abstract"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["reg_num"] = result["reg_num"].astype(str)
    result["pat_year"] = pd.to_numeric(result["pat_year"], errors="raise").astype(int)
    result["patent_title"] = result["patent_title"].fillna("").astype(str)
    result["patent_abstract"] = result["patent_abstract"].fillna("").astype(str)
    result = result.drop_duplicates(subset=["patent_title", "patent_abstract"], keep="first")
    return result


def minimal_preprocess(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def postprocess_phrase(phrase: Optional[str], sao_type: str, lemma_pipeline, stop_words: set[str]) -> str:
    if not phrase:
        return ""
    document = lemma_pipeline(phrase)
    lemmas = [word.lemma.lower() for sentence in document.sentences for word in sentence.words]
    processed = " ".join(word for word in lemmas if word not in stop_words).strip()
    if not processed or not re.search(r"[a-z]", processed):
        return ""
    alpha_count = len(re.findall(r"[a-z]", processed))
    total_count = len(processed.replace(" ", ""))
    if total_count == 0 or alpha_count / total_count <= 0.5:
        return ""
    if sao_type != "action" and len(processed) <= 4:
        return ""
    return processed


def _patent_records(frame: pd.DataFrame) -> list[tuple[str, str]]:
    texts = []
    for row in frame.itertuples(index=False):
        text = f"{row.patent_title}. {row.patent_abstract}".strip()
        texts.append((row.reg_num, text))
    return texts


def extract_triples(
    input_csv: str | Path,
    output_pkl: str | Path,
    batch_size: int = 1000,
    min_confidence: float = 0.8,
    timeout_ms: int = 120000,
    client_timeout_ms: int = 3600000,
    memory: str = "8G",
) -> list[tuple[str, str, str, str]]:
    frame = validate_patent_csv(pd.read_csv(input_csv))
    records = _patent_records(frame)
    output = Path(output_pkl)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        lemma_pipeline = stanza.Pipeline("en", processors="tokenize,pos,lemma", verbose=False)
    except Exception:
        stanza.download("en", verbose=False)
        lemma_pipeline = stanza.Pipeline("en", processors="tokenize,pos,lemma", verbose=False)

    stop_words = _load_stopwords()
    triples: list[tuple[str, str, str, str]] = []

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        with CoreNLPClient(
            annotators=["tokenize", "ssplit", "pos", "lemma", "depparse", "natlog", "openie", "coref"],
            timeout=client_timeout_ms,
            memory=memory,
            properties={
                "openie.resolve_coref": "true",
                "openie.triple.strict": "true",
                "openie.affinity_probability_cap": "0.5",
            },
            be_quiet=True,
        ) as client:
            for reg_num, raw_text in tqdm(batch, desc=f"SAO batch {start // batch_size + 1}"):
                text = minimal_preprocess(raw_text)
                if not text:
                    continue
                try:
                    annotation = client.annotate(text, properties={"timeout": str(timeout_ms)})
                except Exception as error:
                    LOGGER.warning("Skipping patent %s: %s", reg_num, error)
                    continue

                patent_triples: set[tuple[str, str, str, str]] = set()
                for sentence in annotation.sentence:
                    for triple in sentence.openieTriple:
                        if triple.confidence < min_confidence:
                            continue
                        subject = postprocess_phrase(triple.subject, "subject", lemma_pipeline, stop_words)
                        action = postprocess_phrase(triple.relation, "action", lemma_pipeline, stop_words)
                        object_ = postprocess_phrase(getattr(triple, "object", None), "object", lemma_pipeline, stop_words)
                        if subject and action and object_:
                            patent_triples.add((reg_num, subject, action, object_))
                triples.extend(sorted(patent_triples))

        with output.open("wb") as file:
            pickle.dump(triples, file)
        LOGGER.info("Checkpoint saved: %s (%d triples)", output, len(triples))

    return triples
