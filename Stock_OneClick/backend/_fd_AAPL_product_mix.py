"""_fd_AAPL_product_mix.py -- parse the 10-Q's disaggregated-revenue table so the +16.4% quarterly
acceleration can be attributed to specific product lines.

WHY NOT companyfacts: the XBRL companyfacts API returns only CONSOLIDATED values. Product-line
revenue (iPhone / Mac / iPad / Wearables / Services) is tagged with a ProductOrService axis, which
that endpoint drops entirely. The R-files rendered from the same instance document DO carry the
dimensional breakdown, so they are the primary source for the mix question.
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 200)

for f, label in [("_fd_AAPL_q3fy26_R28.html", "Revenue - Disaggregated Net Sales (Q3 FY2026 10-Q)"),
                 ("_fd_AAPL_q3fy26_R46.html", "Segment Information by Reportable Segment (Q3 FY2026 10-Q)")]:
    print("\n" + "=" * 150)
    print(f"{label}   accn 0000320193-26-000020")
    print("=" * 150)
    tables = pd.read_html(HERE / f)
    for i, t in enumerate(tables):
        t = t.dropna(how="all").dropna(axis=1, how="all")
        print(f"\n--- table {i}  shape={t.shape}")
        print(t.to_string(max_colwidth=48))
