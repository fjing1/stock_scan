#!/usr/bin/env python
"""
REVENUE-LEVEL TEST: Arm quarterly royalty revenue vs Apple quarterly iPhone net sales.

SOURCES (all primary):
  Arm royalty revenue  -> Form 6-K Exhibit 99.2 quarterly shareholder letters
                          (files _arm_sl_fye*.txt in this dir; each states
                           "Royalty revenue | <cur> | <prior yr> | <y/y%>")
  Apple iPhone net sales -> Apple Form 10-Q / 10-K, "Products and Services Performance"
                          (files _arm_aapl_2*.txt in this dir)
  FQ4 iPhone derived as FY total minus first-nine-months total from the 10-K.
"""
import pandas as pd, numpy as np

pd.set_option('display.width', 220)

# Arm fiscal quarters end Mar/Jun/Sep/Dec 31. Label by calendar quarter end.
arm_royalty = {  # $m, from 6-K Ex-99.2
    '2023Q2': 400,   # Jun-2023 quarter, from FY25Q1 letter prior-year column
    '2023Q3': 418,   # Sep-2023, from FY25Q2 letter prior-year column
    '2023Q4': 470,   # Dec-2023, FY24Q3 letter
    '2024Q1': 514,   # Mar-2024, FY24Q4 letter
    '2024Q2': 467,   # Jun-2024, FY25Q1 letter
    '2024Q3': 514,   # Sep-2024, FY25Q2 letter
    '2024Q4': 580,   # Dec-2024, FY25Q3 letter
    '2025Q1': 607,   # Mar-2025, FY25Q4 letter
    '2025Q2': 585,   # Jun-2025, FY26Q1 letter
    '2025Q3': 620,   # Sep-2025, FY26Q2 letter
    '2025Q4': 737,   # Dec-2025, FY26Q3 letter
    '2026Q1': 671,   # Mar-2026, FY26Q4 letter
    '2026Q2': 715,   # Jun-2026, FY27Q1 letter
}

# Apple iPhone net sales $m, fiscal quarters ending late Dec/Mar/Jun/Sep -> map to cal quarter
aapl_iphone = {
    '2022Q4': 65775,   # FQ1'23, from 10-Q Dec-2023 prior-year col
    '2023Q1': 51334,   # FQ2'23, from 10-Q Mar-2024 prior-year col
    '2023Q2': 39669,   # FQ3'23, from 10-Q Jun-2024 prior-year col
    '2023Q3': 43805,   # FQ4'23 = FY23 200,583 - 9M 156,778 (10-K FY24 / 10-Q Jun-24)
    '2023Q4': 69702,   # FQ1'24
    '2024Q1': 45963,   # FQ2'24
    '2024Q2': 39296,   # FQ3'24
    '2024Q3': 46222,   # FQ4'24 = FY24 201,183 - 9M 154,961
    '2024Q4': 69138,   # FQ1'25
    '2025Q1': 46841,   # FQ2'25
    '2025Q2': 44582,   # FQ3'25
    '2025Q3': 49025,   # FQ4'25 = FY25 209,586 - 9M 160,561
    '2025Q4': 85269,   # FQ1'26
    '2026Q1': 56994,   # FQ2'26
    '2026Q2': 54252,   # FQ3'26
}

df = pd.DataFrame({'arm_royalty_m': pd.Series(arm_royalty),
                   'aapl_iphone_m': pd.Series(aapl_iphone)}).sort_index()
df['arm_yoy'] = pd.Series(arm_royalty).sort_index().pct_change(4) * 100
df['iph_yoy'] = pd.Series(aapl_iphone).sort_index().pct_change(4) * 100
df['arm_qoq'] = pd.Series(arm_royalty).sort_index().pct_change() * 100
df['iph_qoq'] = pd.Series(aapl_iphone).sort_index().pct_change() * 100
print('=== Arm royalty revenue vs Apple iPhone net sales, by calendar quarter')
print(df.round(1).to_string())

sub = df.dropna(subset=['arm_yoy', 'iph_yoy'])
print(f'\nYoY growth pairs available: n={len(sub)}  ({sub.index[0]} .. {sub.index[-1]})')
print('  corr(YoY royalty growth, YoY iPhone growth) =', round(sub.arm_yoy.corr(sub.iph_yoy), 3))
sub2 = df.dropna(subset=['arm_qoq', 'iph_qoq'])
lv = df.dropna(subset=['arm_royalty_m', 'aapl_iphone_m'])
print('  corr(QoQ royalty growth, QoQ iPhone growth) =', round(sub2.arm_qoq.corr(sub2.iph_qoq), 3), f'n={len(sub2)}')
print('  corr(levels)                                =', round(lv.arm_royalty_m.corr(lv.aapl_iphone_m), 3), f'n={len(lv)}')

# lead/lag: does iPhone growth lead royalty growth by a quarter?
for k in [-2, -1, 0, 1, 2]:
    a = df['arm_yoy']
    b = df['iph_yoy'].shift(k)
    j = pd.concat([a, b], axis=1).dropna()
    print(f'  corr(arm_yoy_t, iph_yoy_t-{k:+d}) = {j.iloc[:,0].corr(j.iloc[:,1]):+.3f}  n={len(j)}')

print('\n=== The decoupling quarters (Apple iPhone accelerating, Arm royalty not)')
print(df.loc['2025Q3':, ['arm_royalty_m', 'arm_yoy', 'aapl_iphone_m', 'iph_yoy']].round(1).to_string())

print("""
NOTE on scale: Arm's own FY2026 20-F states royalty revenue from mobile applications
processors was ~43% of TOTAL royalty revenue in FY ended 2026-03-31. FY26 royalty
revenue was $2,610m (6-K Ex-99.2, Q4 FYE26 letter) and total revenue $4,920m (20-F).
=> mobile AP royalty ~= 0.43 * 2610 = $1,122m = 22.8% of Arm total revenue,
   and Apple is one of MANY mobile AP licensees inside that 22.8%.""")
df.to_csv('_arm_rev_vs_iphone.csv')
