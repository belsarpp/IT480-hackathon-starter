# hackathon.py - Site Safety Monitor (IT 480 Midterm Hackathon)
#
# The client's grader imports this file and calls the functions below.
# Keep the function names, inputs and outputs exactly as specified.
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights" / "best.pt"   # do not change: grading does not run from your repo folder

_model = None


def _get_model():
    """Load the model once and reuse it on every call."""
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHTS))
    return _model


def flag_violations(image_path):
    """
    Return one entry per worker found in the image:
        [{"box": [x1, y1, x2, y2], "no_helmet": bool, "no_vest": bool}, ...]

    box: pixel coordinates (top-left x, y and bottom-right x, y), as Python floats.
    no_helmet, no_vest: Python bools (True or False).
    Return [] if no workers are found.
    """
    raise NotImplementedError("Implement flag_violations")
