"""Synthetic data generator for offline demos and tests.

We plant a *genuine* systematic failure so the algorithms have something real to
find: a "bicycle obscured by a person" slice whose error rate rises with the
degree of occlusion. Ground-truth visual tags are encoded into the synthetic
image URI (see :mod:`slicelens.backends.mock`), so the mock VLM produces a
confidence that correlates with the planted failure mode.

Scenarios (for a bicycle-detection task):

================  ===========================  ==========  ==========
scenario          tags                          count       error prob
================  ===========================  ==========  ==========
clear bicycle     bicycle                       150         0.10
bicycle + person  bicycle, person                90         0.40
obscured bicycle  bicycle, person, obscured      60         0.80  <- GT slice
motorcycle (FP)   motorcycle, wheels             30         0.50
================  ===========================  ==========  ==========
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .data_types import ErrorType, Instance, Region

# (name, tags, count, error_prob, error_type)
_SCENARIOS: Sequence[tuple[str, list[str], int, float, ErrorType]] = (
    ("clear", ["bicycle"], 150, 0.10, ErrorType.FALSE_NEGATIVE),
    ("near_person", ["bicycle", "person"], 90, 0.40, ErrorType.FALSE_NEGATIVE),
    ("obscured", ["bicycle", "person", "obscured"], 60, 0.80, ErrorType.FALSE_NEGATIVE),
    ("motorcycle", ["motorcycle", "wheels"], 30, 0.50, ErrorType.FALSE_POSITIVE),
)

# The ground-truth error slice we expect SliceLens to recover.
GT_SLICE_NAME = "bicycle obscured by a person"
GT_SCENARIO = "obscured"


def make_synthetic_detection_instances(
    seed: int = 0,
) -> tuple[list[Instance], set[str]]:
    """Return (instances, ground_truth_instance_ids_for_the_obscured_slice)."""
    rng = random.Random(seed)
    instances: list[Instance] = []
    gt_ids: set[str] = set()
    idx = 0
    for name, tags, count, err_prob, err_type in _SCENARIOS:
        for _ in range(count):
            inst_id = f"inst_{idx}"
            tag_str = "|".join(tags)
            image_path = f"synthetic://{inst_id}?tags={tag_str}"
            # Random but plausible bounding box.
            x0 = rng.uniform(0, 400)
            y0 = rng.uniform(0, 300)
            bbox = (x0, y0, x0 + rng.uniform(40, 200), y0 + rng.uniform(40, 200))
            is_error = rng.random() < err_prob
            instances.append(
                Instance(
                    instance_id=inst_id,
                    image_path=image_path,
                    region=Region(bbox=bbox),
                    label="bicycle" if "bicycle" in tags else "motorcycle",
                    is_error=is_error,
                    error_type=err_type if is_error else ErrorType.CORRECT,
                    metadata={"scenario": name, "tags": tags},
                )
            )
            if name == GT_SCENARIO:
                gt_ids.add(inst_id)
            idx += 1
    rng.shuffle(instances)
    return instances, gt_ids
