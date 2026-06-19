from patent_sao.dataset import _sample_random_negatives
import random


def test_negative_sampling_excludes_existing():
    existing = {"s1;a1;o1"}
    result = _sample_random_negatives(existing, ["s1", "s2"], ["a1;o1", "a2;o2"], 2, random.Random(42))
    assert len(result) == 2
    assert all(";".join(item) not in existing for item in result)
