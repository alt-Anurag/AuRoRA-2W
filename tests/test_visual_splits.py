from scripts.audit_visual_splits import cross_split_pairs
from scripts.prepare_rdd_holdout import components, assign_groups


def test_perceptual_chain_stays_one_group_with_exact_duplicates():
    # A-B and B-C are close; A-C is not. Grouping must be transitive.
    features = [(0, 0, "a"), (3, 3, "b"), (15, 15, "c"), (255, 255, "a")]
    groups, _ = components(features, phash_limit=2, dhash_limit=2)
    assert groups == [[0, 1, 2, 3]]


def test_cross_split_rule_requires_both_hashes_and_never_compares_within_split():
    records = [{"split": s} for s in ("train", "train", "val", "test")]
    features = [(0, 0, "a"), (0, 0, "a"), (1, 1, "v"), (1, 1023, "t")]
    pairs = list(cross_split_pairs(records, features, phash_limit=2, dhash_limit=2))
    assert {(p["a"], p["b"]) for p in pairs} == {(0, 2), (1, 2)}
    assert not any(p["identical_decoded_pixels"] for p in pairs)


def test_partition_assignment_preserves_groups_and_presence_strata():
    groups = [[0, 1], [2], [3], [4, 5], [6], [7]]
    records = [{"boxes": [1] if i < 4 else []} for i in range(8)]
    val = assign_groups(groups, records, .2, 42)
    assert val == assign_groups(groups, records, .2, 42)
    assert any(records[i]["boxes"] for i in val)
    assert any(not records[i]["boxes"] for i in val)
    for group in groups:
        assert len(set(group) & val) in (0, len(group))



def test_same_split_cross_source_rule_remaps_indices_and_excludes_within_source():
    from scripts.audit_visual_splits import same_split_cross_source_pairs
    records = [{"source": src, "split": split} for src, split in
               [("seg", "val"), ("det", "train"), ("seg", "train"), ("seg", "train"), ("det", "val")]]
    features = [(0, 0, str(i)) for i in range(5)]
    pairs = list(same_split_cross_source_pairs(records, features))
    assert {frozenset((p["a"], p["b"])) for p in pairs} == {frozenset((1, 2)), frozenset((1, 3)), frozenset((0, 4))}


def test_combined_review_has_no_repeated_pairs_and_keeps_both_hash_thresholds():
    from scripts.audit_visual_splits import review_pairs
    records = [{"source": src, "split": split} for src, split in
               [("seg", "train"), ("det", "train"), ("det", "val"), ("rdd", "train")]]
    features = [(0, 0, "a"), (1, 1, "b"), (0, 0, "a"), (1, 1023, "bad")]
    pairs = list(review_pairs(records, features, include_cross_source=True))
    keys = [frozenset((p["a"], p["b"])) for p in pairs]
    assert len(keys) == len(set(keys)) == 3
    assert frozenset((0, 1)) in keys and frozenset((0, 2)) in keys and frozenset((1, 2)) in keys
    assert len(list(review_pairs(records, features))) == 2
