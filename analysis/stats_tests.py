#!/usr/bin/env python3
"""
Paired statistical analysis across the four pipelines (per-case Dice).
Wilcoxon signed-rank tests + failed-case (<0.8) overlap.

Run (TWCC):
    /opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python stats_tests.py
"""
import csv
import os
import statistics as st
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def load(path, dk):
    d = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, path))):
        cid = r.get("Case_ID") or r.get("case")
        d[cid] = float(r[dk])
    return d


C = load("classical.csv", "dice")
D3 = load("unet3d.csv", "Dice_Score")
D25 = load("unet25d.csv", "Dice_Score")
H = load("hybrid.csv", "Dice_Score")

common = sorted(set(C) & set(D3) & set(D25) & set(H), key=int)
vec = lambda d: [d[c] for c in common]
c, d3, d25, h = vec(C), vec(D3), vec(D25), vec(H)
print(f"matched cases: {len(common)}")
for name, v in [("Classical", c), ("3D", d3), ("2.5D", d25), ("Hybrid", h)]:
    print(f"  {name:9s} mean={st.mean(v):.4f} median={st.median(v):.4f} "
          f"<0.8: {sum(x < 0.8 for x in v)}")

print("\nWilcoxon signed-rank (paired, two-sided):")
pairs = [(h, d3, "Hybrid", "3D"), (d3, d25, "3D", "2.5D"),
         (h, d25, "Hybrid", "2.5D"), (d3, c, "3D", "Classical"),
         (h, c, "Hybrid", "Classical"), (d25, c, "2.5D", "Classical")]
def cliffs_delta(a, b):
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def cliff_mag(d):
    ad = abs(d)
    return ("negligible" if ad < 0.147 else "small" if ad < 0.33
            else "medium" if ad < 0.474 else "large")


def rank_biserial(a, b):
    diffs = [x - y for x, y in zip(a, b) if x - y != 0]
    ranks = stats.rankdata([abs(x) for x in diffs])
    wp = sum(r for r, x in zip(ranks, diffs) if x > 0)
    wn = sum(r for r, x in zip(ranks, diffs) if x < 0)
    return (wp - wn) / (wp + wn)

for a, b, an, bn in pairs:
    w, p = stats.wilcoxon(a, b)
    cd = cliffs_delta(a, b)
    print(f"  {an:9s} vs {bn:9s}: W={w:.1f}  p={p:.3g}  "
          f"Cliff_d={cd:+.3f} ({cliff_mag(cd)})  r_rb={rank_biserial(a, b):+.3f}")

fc = {n: set(cc for cc, x in zip(common, v) if x < 0.8)
      for n, v in [("3D", d3), ("2.5D", d25), ("Hybrid", h)]}
print("\nFailed cases (<0.8):")
for n, s in fc.items():
    print(f"  {n}: {sorted(s, key=int)}")
print(f"  3D == Hybrid failed set? {fc['3D'] == fc['Hybrid']}")
print(f"  2.5D disjoint from 3D/Hybrid? {fc['2.5D'].isdisjoint(fc['3D'])}")
