"""
build_image_report.py
---------------------
Generates report-grade figures and an Excel workbook for the porous-flow dataset.

Inputs (from --data-dir, default: Datasets/):
  • den.npy  : (N, H, W[,1]) circle-pack mask per sample (0=solid, 1=void). Optional.
  • vx.npy   : (N, H, W[,1]) x-component of velocity (lattice units).
  • vy.npy   : (N, H, W[,1]) y-component of velocity (lattice units).

Outputs (to --out-xlsx, default: Results/flow_summary.xlsx):
  • Excel workbook with:
      - Summary          : dataset overview and global stats
      - PerImageStats    : one row per sample (porosity, vx/vy/|v| stats)
      - S01..S10         : first 10 samples; each sheet embeds its figure and
                           includes a summary block plus full v_x, v_y, |v| matrices.
    Note: matrices are written scaled by ×10^k for readability; the applied scale
    is noted on each sheet. The |v| color scale is consistent (global 99th pct).
  • Results/figs/sample_XX.png: paired “Input” (mask) and “Predicted |v|” figure per sample.

Requirements:
  numpy, pandas, matplotlib, openpyxl, pillow

Quick install:
  python -m pip install numpy pandas matplotlib openpyxl pillow

Run:
  python Scripts/build_image_report.py --data-dir Datasets --out-xlsx Results/flow_summary.xlsx
"""
