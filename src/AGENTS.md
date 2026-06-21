# src/AGENTS.md - Source Package Engineering Specifications

This protocol defines the implementation guidelines, safety red-lines, and configuration thresholds for `dataset`, `gatekeeper`, and `utils` modules.
## 0. PACKAGE STRUCTURE & MODULARITY

Per project-wide modularity rules (`rules.md` §6), the `src/` directory is organized into single-responsibility packages:

| Package | Responsibility | Key Files |
|:---|:---|:---|
| `src/dataset/` | Data loading, augmentation | `loader.py`, `augmentation.py` |
| `src/gatekeeper/` | Image authenticity verification (inference-time) | `authenticity.py`, `deduplication.py`, `tampering.py` |
| `src/utils/` | Visualization, report generation, Grad-CAM | `report_generator.py`, `grad_cam.py` |
| `src/training/` | Training loop, stage management, authenticity classifier | `trainer.py`, `stages.py`, `authenticity.py` |
| `src/models/` | Neural network architectures & losses | See `src/models/AGENTS.md` |

**Rule**: New functionality MUST be added to the appropriate package based on responsibility. Do NOT create monolithic files spanning multiple domains.

---

---

## 1. src/dataset/ → See `src/dataset/AGENTS.md`

*Full specifications (static normalization red-line, post-hoc clinical audit gates, augmentation rules) are in [`src/dataset/AGENTS.md`](src/dataset/AGENTS.md).*

---

## 2. src/gatekeeper/ → See `src/gatekeeper/AGENTS.md`

*Full specifications (authenticity classifier, pHash + cosine dedup, tampering detection) are in [`src/gatekeeper/AGENTS.md`](src/gatekeeper/AGENTS.md).*

---

## 3. src/utils/ SPECIFICATIONS (Explainability & Clinical Reports)

- **TI-RADS Point System Contract**: The `calculate_tirads_points` mapping table (composition, echogenicity, shape, margin, foci points) must adhere strictly to the official clinical Guidelines:
  - **Composition**: Cystic/Spongiform ($0$ pts), Mixed ($1$ pt), Solid ($2$ pts).
  - **Echogenicity**: Anechoic ($0$ pts), Iso/Hyperechoic ($1$ pt), Hypoechoic ($2$ pts), Very Hypoechoic ($3$ pts).
  - **Shape**: Wider-than-tall ($0$ pts), Taller-than-wide ($3$ pts).
  - **Margin**: Smooth ($0$ pts), Ill-defined ($0$ pts), Lobulated ($2$ pts), Extrathyroidal extension ($3$ pts).
  - **Foci**: None ($0$ pts), Macrocalcifications ($1$ pt), Peripheral ($2$ pts), Punctate microcalcifications ($3$ pts).
- **Report Consistency**: The clinical report generator must strictly output text matching these points and the corresponding risk stratification categories.
