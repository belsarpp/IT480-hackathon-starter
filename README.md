# IT 480 Midterm Hackathon: Site Safety Monitor

**The full instructions, schedule and rubric are on Canvas (Midterm Hackathon instruction manual).** 

## Set up your team repository (first 15 minutes)

1. Click **Use this template > Create a new repository**. Do not fork this repository.
2. Name it `IT480-hackathon-Team<number>` (for example, `IT480-hackathon-Team03`).
3. Set it to **Private**.
4. In **Settings > Collaborators**, add your three teammates and `belsarpp`.
5. Submit the repository link on Canvas (*Team Repository Link*).

## What's in this repository

| File | Purpose |
|---|---|
| `hackathon.py` | Stub for the required function. Keep the name, input and output exactly as specified. |
| `sample_grader.py` | Checks that your code meets the interface (not accuracy) |
| `sample_images/` | 3 images used by the sample grader |
| `weights/` | Put your trained `best.pt` here |
| `notebooks/Hackathon_Team_Notebook.ipynb` | Your team's working notebook |

## Technical constraints

| Setting | Requirement |
|---|---|
| Dataset | `construction-ppe.yaml` (downloads automatically) |
| Model | `yolo26n.pt` (nano) |
| Image size | 640 × 640 |
| Epochs | 20–50 |
| Packages | `ultralytics`, `numpy`, `opencv-python`, and the Python standard library only |
| Runtime | Must run on a Colab T4 GPU |

## Interface

```python
def flag_violations(image_path):
    # Returns one entry per worker:
    # [{"box": [x1, y1, x2, y2], "no_helmet": True/False, "no_vest": True/False}, ...]
    # box: pixel coordinates. Return [] if no workers are found.
```

`hackathon.py` must load `weights/best.pt` with a path relative to `hackathon.py`, as in the stub.

## Required layout at the deadline

```
your-team-repo/
├── hackathon.py
├── weights/best.pt
├── ESTIMATE.md
├── README.md
└── notebooks/
```

## Sample grader

Run in Colab from your repository folder:

```
!python sample_grader.py --team-dir .
```

Code that fails the sample grader receives 0 points for components A and B.

---

# Your team README (replace everything above by Wed Nov 4, 11:59 pm)

## Team
## Approach
## Threshold
## Results
## Real-world testing
## Day 2 requirement
## Known limitations
## AI log
