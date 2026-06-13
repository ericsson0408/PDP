#!/usr/bin/env python3
# Professor critiques #1 (Dice on full-120 includes training data) and #2 (no
# timing dispersion). Computes, from the archived per-case CSVs:
#   #1  the held-out 3D/hybrid validation split (exactly reproduced: sorted
#       AIIB23_*.nii.gz filenames -> random.Random(42).shuffle -> first 12, the
#       same recipe as training/3d_dynunet/train.py), then val-only mean Dice +
#       paired Wilcoxon / Cliff's delta on those held-out cases.
#   #2  mean / std / 95% CI of per-case total time for every pipeline.
#
#   /opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python analysis/stats_heldout_timing.py
import csv, os, random, statistics as st
from scipy import stats

A = os.path.join(os.path.dirname(__file__), "..", "results")
def load(path, idk, vks):
    out = {}
    for r in csv.DictReader(open(os.path.join(A, path))):
        cid = r.get("Case_ID") or r.get("case")
        out[cid] = {k: float(r[k]) for k in vks if k in r}
    return out

dice = {
    "Classical": {c: v["dice"]        for c, v in load("classical.csv", "case", ["dice"]).items()},
    "3D":        {c: v["Dice_Score"]  for c, v in load("unet3d.csv", "Case_ID", ["Dice_Score"]).items()},
    "2.5D":      {c: v["Dice_Score"]  for c, v in load("unet25d.csv", "Case_ID", ["Dice_Score"]).items()},
    "Hybrid":    {c: v["Dice_Score"]  for c, v in load("hybrid.csv", "Case_ID", ["Dice_Score"]).items()},
}
total = {
    "Classical": {c: v["total_s"]    for c, v in load("classical.csv", "case", ["total_s"]).items()},
    "3D":        {c: v["Total_Time"] for c, v in load("unet3d.csv", "Case_ID", ["Total_Time"]).items()},
    "2.5D":      {c: v["Total_Time"] for c, v in load("unet25d.csv", "Case_ID", ["Total_Time"]).items()},
    "Hybrid":    {c: v["Total_Time"] for c, v in load("hybrid.csv", "Case_ID", ["Total_Time"]).items()},
}

# ---- #1 reproduce the 3D/hybrid held-out split -----------------------------
ids = sorted(dice["3D"].keys(), key=int)
fnames = sorted(f"AIIB23_{c}.nii.gz" for c in ids)        # == sorted(glob(*.nii.gz))
order  = [f.split("_")[1].split(".")[0] for f in fnames]  # case-id order pre-shuffle
rng = random.Random(42); rng.shuffle(order)
n_val = max(1, round(len(order) * 0.10))                  # val_frac=0.1 -> 12
val = set(order[:n_val])
print(f"[#1] 3D/hybrid held-out val ({len(val)} cases): {sorted(val, key=int)}\n")

def cliffs(a, b):
    gt = sum(x > y for x in a for y in b); lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))

def compare(name_a, name_b, cases):
    cs = sorted(cases, key=int)
    a = [dice[name_a][c] for c in cs]; b = [dice[name_b][c] for c in cs]
    try: p = stats.wilcoxon(a, b).pvalue
    except Exception: p = float("nan")
    d = cliffs(a, b)
    print(f"  {name_a:8s} vs {name_b:8s} | n={len(cs):3d} "
          f"mean {st.mean(a):.4f} vs {st.mean(b):.4f} | Wilcoxon p={p:.2e} Cliff d={d:+.3f}")

allc = sorted(set(dice["3D"]) & set(dice["2.5D"]) & set(dice["Hybrid"]) & set(dice["Classical"]), key=int)
print("[#1] FULL set (120, includes training):")
for a, b in [("Hybrid","3D"),("3D","2.5D"),("3D","Classical")]: compare(a,b,allc)
print(f"\n[#1] HELD-OUT 3D/hybrid val ({len(val)} cases, unbiased for 3D & hybrid):")
for a, b in [("Hybrid","3D"),("3D","2.5D"),("3D","Classical")]: compare(a,b,val)
print("  (held-out val mean Dice per model:)")
for m in ["3D","Hybrid","2.5D","Classical"]:
    v=[dice[m][c] for c in sorted(val,key=int)]; print(f"     {m:9s} {st.mean(v):.4f}")

# ---- #2 timing dispersion --------------------------------------------------
print("\n[#2] per-case total time: mean / std / 95% CI (full 120):")
print(f"  {'pipeline':9s} {'mean_s':>7} {'std_s':>7} {'95%CI':>16} {'min':>6} {'max':>6}")
for m in ["Classical","3D","Hybrid","2.5D"]:
    v = [total[m][c] for c in sorted(total[m], key=int)]
    mean, sd, n = st.mean(v), st.pstdev(v), len(v)
    half = 1.96 * sd / (n ** 0.5)
    print(f"  {m:9s} {mean:7.2f} {sd:7.2f}  [{mean-half:5.2f}, {mean+half:5.2f}]  {min(v):6.2f} {max(v):6.2f}")
