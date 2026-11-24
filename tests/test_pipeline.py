"""Integration test: the full mock pipeline recovers the planted slice."""

from slicelens import SliceLens, SliceLensConfig
from slicelens.backends import MockLLM, MockVLM
from slicelens.evaluation import precision_at_k
from slicelens.synthetic import GT_SCENARIO, make_synthetic_detection_instances


def _run():
    instances, gt_ids = make_synthetic_detection_instances(seed=0)
    config = SliceLensConfig(task_type="detection", grounding="bbox")
    sl = SliceLens(MockLLM(), MockVLM(logit_scale=2.5, noise=0.15), config)
    result = sl.run(instances, label="bicycle")
    return result, gt_ids


def test_generates_knowledge_and_data_hypotheses():
    result, _ = _run()
    sources = {h.source for h in result.hypotheses}
    assert "knowledge" in sources
    assert "data" in sources


def test_finds_exactly_the_planted_slice():
    result, _ = _run()
    systematic_queries = [es.hypothesis.query for es in result.error_slices]
    assert "bicycle obscured by a person" in systematic_queries
    # Flat / unrelated hypotheses must not be flagged.
    assert "a very small bicycle" not in systematic_queries


def test_planted_slice_has_high_precision():
    result, gt_ids = _run()
    target = next(
        es for es in result.error_slices
        if es.hypothesis.query == "bicycle obscured by a person"
    )
    assert precision_at_k(target.candidate, gt_ids, k=10) >= 0.8


def test_obscured_instances_are_high_error():
    instances, gt_ids = make_synthetic_detection_instances(seed=0)
    obscured = [i for i in instances if i.metadata["scenario"] == GT_SCENARIO]
    rate = sum(i.is_error for i in obscured) / len(obscured)
    assert rate > 0.6  # planted error prob is 0.8
