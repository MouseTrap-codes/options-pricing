from typing import List, Tuple

from .bt_euro_test_inputs import (
    EUROPEAN_CALL_INPUTS,
    EUROPEAN_EDGE_CASES,
    EUROPEAN_PUT_INPUTS,
)

AMERICAN_CALL_INPUTS: List[Tuple[str, dict]] = [
    (name.replace("european", "american"), {**params, "exercise_type": "american"})
    for name, params in EUROPEAN_CALL_INPUTS
]

AMERICAN_PUT_INPUTS: List[Tuple[str, dict]] = [
    (name.replace("european", "american"), {**params, "exercise_type": "american"})
    for name, params in EUROPEAN_PUT_INPUTS
]

AMERICAN_EDGE_CASES: List[Tuple[str, dict]] = [
    (name.replace("european", "american"), {**params, "exercise_type": "american"})
    for name, params in EUROPEAN_EDGE_CASES
]
