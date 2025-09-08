"""
build_image_report.py  &  build_input_report.py
------------------------------------------------
Tools to generate report-grade figures and Excel workbooks for the porous-flow dataset.

DATA LAYOUT (from --data-dir, default: Datasets/)
  • den.npy  : (N, H, W[,1]) circle-pack mask per sample (0=solid, 1=void). Optional for image report.
  • vx.npy   : (N, H, W[,1]) x-component of velocity (lattice units).
  • vy.npy   : (N, H, W[,1]) y-component of velocity (lattice units).

----------------------------------------------------------------------
A) build_image_report.py  — Input + Predicted |v| report (like the paper)
----------------------------------------------------------------------
Purpose
  Paired figures (“Input” mask on the left, “Predicted |v|” on the right),
  plus an Excel workbook with summary stats and per-sample pages.

Outputs (default root: Results/flow_report/)
  • flow_summary.xlsx with sheets:
      - Summary          : dataset overview and global stats
      - PerImageStats    : one row per sample (porosity, vx/vy/|v| stats)
      - S01..S10         : first 10 samples; each sheet embeds its figure and
                           includes a summary block plus full v_x, v_y, |v| matrices.
        Note: matrices are written scaled by ×10^k for readability; the applied
              scale is shown on each sheet. The |v| color scale is consistent
              (global 99th percentile across the dataset).
  • figs/sample_XX.png   : paired “Input” + “Predicted |v|” PNG per sample.

Usage
  python Scripts/build_image_report.py --data-dir Datasets
  # custom output folder (Excel + figs/ inside it):
  python Scripts/build_image_report.py --data-dir Datasets --out-dir Results/flow_report
  # explicit Excel path (absolute or relative):
  python Scripts/build_image_report.py --data-dir Datasets --out-xlsx Results/flow_report/flow_summary.xlsx

Render more than 10 samples?
  The script currently fixes pages to the first 10 samples.
  To increase, edit in the script:
      picks = list(range(min(10, N)))
  -> for 25 pages:
      picks = list(range(min(25, N)))
  (If you want, we can expose a --max-pages flag like in build_input_report.py.)

----------------------------------------------------------------------
B) build_input_report.py  — Input-only report (no velocities/predictions)
----------------------------------------------------------------------
Purpose
  Load ONLY the input mask (den.npy) and render clean input figures +
  a compact Excel you can use later to compare against your own model’s predictions.

Outputs (default root: Results/inputs_report/)
  • input_only_summary.xlsx with sheets:
      - Summary          : dataset overview
      - PerImageStats    : per-sample porosity (and H/W)
      - S01..SXX         : first XX samples; each page embeds the input figure
                           (grayscale 0–1) with the sample’s porosity.
  • figs/sample_XX.png   : input-only mask PNGs.

Usage
  # default destination (Results/inputs_report/)
  python Scripts/build_input_report.py --data-dir Datasets
  # pick a custom output folder (Excel + figs/ inside it):
  python Scripts/build_input_report.py --data-dir Datasets --out-dir Results/InputOnly
  # explicit Excel path (absolute or relative):
  python Scripts/build_input_report.py --data-dir Datasets --out-xlsx Results/InputOnly/input_only_summary.xlsx

Render MORE pages (e.g., 20 or 100)
  Use --max-pages:
    python Scripts/build_input_report.py --data-dir Datasets --max-pages 20
    python Scripts/build_input_report.py --data-dir Datasets --max-pages 100

----------------------------------------------------------------------
Requirements
  python -m pip install numpy pandas matplotlib openpyxl pillow

Common tips
  • When using multi-line shell commands, the backslash "\" must be the LAST character on the line.
  • If you pass --out-xlsx with a relative path, it is placed under --out-dir automatically.
  • Figures always save next to the Excel under a figs/ subfolder.
"""
