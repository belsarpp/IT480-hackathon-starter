"""
IT 480 Midterm Hackathon: sample grader.

Checks that your repository meets the interface specification. It does NOT score
your system; the instructor's grader uses hidden images and the same checks.

Usage (from your repository folder):
    python sample_grader.py --team-dir .

On Day 2, also check the new function:
    python sample_grader.py --team-dir . --day2-function <name announced in class>
"""
import argparse
import ast
import importlib.util
import math
import sysconfig
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ALLOWED_PACKAGES = {"ultralytics", "numpy", "cv2", "torch", "PIL"}  # torch and PIL ship with ultralytics
_STDLIB_DIRS = {os.path.realpath(sysconfig.get_paths()[k]) for k in ("stdlib", "platstdlib")}


def is_stdlib(name):
    """True for Python standard-library modules (works on Python 3.8+)."""
    if name in getattr(sys, "stdlib_module_names", ()) or name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    origin = spec.origin or ""
    if origin in ("built-in", "frozen"):
        return True
    path = os.path.realpath(origin)
    if "site-packages" in path or "dist-packages" in path:
        return False
    return any(path.startswith(d) for d in _STDLIB_DIRS)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SLOW_SECONDS = 10.0
REQUIRED_KEYS = {"box", "no_helmet", "no_vest"}

errors, warnings = [], []


def fail(msg):
    errors.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  WARN  {msg}")


def ok(msg):
    print(f"  OK    {msg}")


def check_files(team_dir):
    print("\n[1] Repository files")
    hk = team_dir / "hackathon.py"
    w = team_dir / "weights" / "best.pt"
    if hk.exists():
        ok("hackathon.py found")
    else:
        fail("hackathon.py is missing from the repository root")
    if w.exists():
        ok(f"weights/best.pt found ({w.stat().st_size / 1e6:.1f} MB)")
    else:
        fail("weights/best.pt is missing (upload your trained best.pt into weights/)")
    if (team_dir / "ESTIMATE.md").exists():
        ok("ESTIMATE.md found")
    else:
        warn("ESTIMATE.md is missing (required by the hard deadline)")
    return hk.exists()


def check_imports(hk_path):
    print("\n[2] Imports in hackathon.py")
    try:
        tree = ast.parse(hk_path.read_text())
    except SyntaxError as e:
        fail(f"hackathon.py has a syntax error: line {e.lineno}: {e.msg}")
        return False
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    outside = sorted(m for m in found if m not in ALLOWED_PACKAGES and not is_stdlib(m))
    if outside:
        fail(f"packages outside the allowed list: {', '.join(outside)} "
             "(allowed: ultralytics, numpy, opencv-python, plus the Python standard library)")
    else:
        ok("only allowed packages are imported")
    if "/content/" in hk_path.read_text() or "C:\\" in hk_path.read_text():
        warn("hackathon.py contains an absolute path (/content/... or C:\\...). "
             "Load files relative to hackathon.py, as in the starter stub.")
    return True


def check_environment():
    print("\n[3] Python environment")
    ok(f"Python {sys.version.split()[0]} at {sys.executable}")
    missing = [p for p in ("ultralytics", "numpy", "cv2") if importlib.util.find_spec(p) is None]
    if missing:
        pip_names = {"cv2": "opencv-python"}
        fail("not installed in this Python: " + ", ".join(missing) + ". Install with: "
             f"{Path(sys.executable).name} -m pip install " + " ".join(pip_names.get(m, m) for m in missing)
             + "  (or run this check in Colab, where they are available)")
        return False
    ok("ultralytics, numpy and opencv are installed")
    return True


def load_module(team_dir):
    print("\n[4] Importing hackathon.py from a different working directory")
    # Grading does not run inside your repository folder, so relative paths must
    # be built from the location of hackathon.py, not from the current directory.
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    sys.path.insert(0, str(team_dir))
    try:
        spec = importlib.util.spec_from_file_location("team_hackathon", team_dir / "hackathon.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok("hackathon.py imported")
        return mod
    except Exception as e:
        fail(f"importing hackathon.py raised {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return None
    finally:
        os.chdir(cwd)


def tname(v):
    t = type(v)
    return t.__name__ if t.__module__ == "builtins" else f"{t.__module__.split('.')[0]}.{t.__name__}"


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_output(out, img_path):
    try:
        import cv2
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2] if im is not None else (None, None)
    except Exception:
        h = w = None
    name = img_path.name
    if not isinstance(out, list):
        fail(f"{name}: flag_violations must return a list, got {tname(out)}")
        return
    for i, entry in enumerate(out):
        tag = f"{name}, worker {i}"
        if not isinstance(entry, dict):
            fail(f"{tag}: each entry must be a dict, got {type(entry).__name__}")
            continue
        keys = set(entry)
        if keys != REQUIRED_KEYS:
            missing, extra = REQUIRED_KEYS - keys, keys - REQUIRED_KEYS
            fail(f"{tag}: keys must be exactly {sorted(REQUIRED_KEYS)}"
                 + (f"; missing {sorted(missing)}" if missing else "")
                 + (f"; unexpected {sorted(extra)}" if extra else ""))
            continue
        box = entry["box"]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            fail(f"{tag}: box must be a list of 4 numbers [x1, y1, x2, y2]")
        elif not all(is_number(v) for v in box):
            types = sorted({tname(v) for v in box})
            fail(f"{tag}: box values must be Python int or float, got {types}. "
                 "Convert with float(v) (numpy and torch types are not accepted).")
        else:
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                fail(f"{tag}: box must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1, got {box}")
            elif w and (max(x1, x2) <= 1.0 and max(y1, y2) <= 1.0):
                fail(f"{tag}: box looks normalized (0-1). Use pixel coordinates.")
            elif w and (x1 < -1 or y1 < -1 or x2 > w + 1 or y2 > h + 1):
                warn(f"{tag}: box {box} extends outside the {w}x{h} image")
        for k in ("no_helmet", "no_vest"):
            if not isinstance(entry[k], bool):
                fail(f"{tag}: {k} must be True or False (Python bool), got {tname(entry[k])}. "
                     "Convert with bool(...).")


def run_function(mod, fn_name, images, checker):
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        fail(f"hackathon.py has no function named {fn_name}()")
        return
    for k, img in enumerate(images):
        t0 = time.time()
        try:
            out = fn(str(img))
        except NotImplementedError:
            fail(f"{fn_name}() is not implemented yet (it raised NotImplementedError)")
            return
        except Exception as e:
            fail(f"{img.name}: {fn_name}() raised {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            continue
        dt = time.time() - t0
        n_before = len(errors)
        checker(out, img)
        if len(errors) == n_before:
            ok(f"{img.name}: {describe(out)} ({dt:.1f}s)")
        if dt > (60.0 if k == 0 else SLOW_SECONDS):   # the first call may include loading the model
            warn(f"{img.name}: took {dt:.1f}s. Load the model once (module level or a cached loader), not on every call.")


def describe(out):
    if isinstance(out, bool):
        return f"returned {out}"
    if isinstance(out, list):
        nh = sum(1 for e in out if isinstance(e, dict) and e.get("no_helmet") is True)
        nv = sum(1 for e in out if isinstance(e, dict) and e.get("no_vest") is True)
        return f"{len(out)} worker(s), {nh} no_helmet, {nv} no_vest"
    return "returned a value"


def check_bool(out, img_path):
    if not isinstance(out, bool):
        fail(f"{img_path.name}: must return True or False (Python bool), got {tname(out)}")


def main():
    ap = argparse.ArgumentParser(description="IT 480 hackathon sample grader")
    ap.add_argument("--team-dir", default=".", help="path to your repository folder")
    ap.add_argument("--images", default=None, help="folder of test images (default: sample_images/)")
    ap.add_argument("--day2-function", default=None, help="name of the Day 2 function, once announced")
    args = ap.parse_args()

    team_dir = Path(args.team_dir).resolve()
    img_dir = Path(args.images).resolve() if args.images else team_dir / "sample_images"
    if not img_dir.exists():
        img_dir = Path(__file__).resolve().parent / "sample_images"
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXT) if img_dir.exists() else []

    print("IT 480 Hackathon sample grader")
    print(f"Team folder: {team_dir}")
    print(f"Images:      {img_dir} ({len(images)} found)")

    if check_files(team_dir) and check_imports(team_dir / "hackathon.py") and check_environment():
        mod = load_module(team_dir)
        if mod is not None:
            if not images:
                fail("no sample images found; pass --images <folder>")
            else:
                print("\n[5] flag_violations() on sample images")
                run_function(mod, "flag_violations", images, check_output)
                if args.day2_function:
                    print(f"\n[6] {args.day2_function}() on sample images")
                    run_function(mod, args.day2_function, images, check_bool)

    print("\n" + "=" * 60)
    if errors:
        print(f"RESULT: FAIL ({len(errors)} problem(s), {len(warnings)} warning(s))")
        print("Code that fails this check receives 0 for components A and B.")
        sys.exit(1)
    print(f"RESULT: PASS ({len(warnings)} warning(s))")
    print("Your code meets the interface. This does not measure accuracy.")


if __name__ == "__main__":
    main()
