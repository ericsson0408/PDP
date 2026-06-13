#!/usr/bin/env python3
"""
一次性把 AIIB23 抽成「單張 2D 切片」存進單一 memmap（uint8，省空間）。
2.5D 的相鄰切片在「訓練時即時組」（讀 z-1,z,z+1 三列），所以這裡每張只存一次，
不存 slab 重疊 → 全 120 case 約 6GB（float16+slab3 版要 36GB，放不下）。

輸出 <out>/：
  X.npy memmap uint8 (N,size,size)  HU clip[-1000,600]→[0,255]
  Y.npy memmap uint8 (N,size,size)  氣道遮罩
  index.csv 每列 (row, case, z, zlocal, has_airway, n_in_case)
            zlocal=此切片在該 case 連續儲存中的序號（供即時取相鄰）
注意：為了能即時取相鄰，這裡存「每個 case 的全部切片」（連續存），不做背景取樣，
      取樣交給訓練時的 sampler（train 才下採樣背景，val 用全部含氣道）。
"""
import argparse, os, glob, csv
import numpy as np, nibabel as nib
import torch, torch.nn.functional as F
HU_LO,HU_HI=-1000.0,600.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--size",type=int,default=256)
    ap.add_argument("--limit-cases",type=int,default=0)
    ap.add_argument("--out",default="data/slices25d")
    a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    imgs=sorted(glob.glob("data/img/AIIB23_*.nii.gz"))
    if a.limit_cases>0: imgs=imgs[:a.limit_cases]

    # pass1：算總切片數（= 每個 case 的所有 z）
    plan=[]; N=0
    for ip in imgs:
        gp=ip.replace("/img/","/gt/")
        if not os.path.exists(gp): continue
        cid=os.path.basename(ip).replace("AIIB23_","").replace(".nii.gz","")
        nz=nib.load(ip).shape[2]
        plan.append((ip,gp,cid,nz)); N+=nz
        print(f"[plan] case {cid}: {nz} slices (累計 {N})",flush=True)
    print(f"[plan] 總 {N} 切片 → memmap uint8 (N,{a.size},{a.size})",flush=True)

    X=np.lib.format.open_memmap(os.path.join(a.out,"X.npy"),mode="w+",dtype=np.uint8,shape=(N,a.size,a.size))
    Y=np.lib.format.open_memmap(os.path.join(a.out,"Y.npy"),mode="w+",dtype=np.uint8,shape=(N,a.size,a.size))
    rows=[]; r=0
    for k,(ip,gp,cid,nz) in enumerate(plan):
        hu=nib.load(ip).get_fdata().astype(np.float32)
        hu=np.clip((np.clip(hu,HU_LO,HU_HI)-HU_LO)/(HU_HI-HU_LO)*255.0,0,255)
        g=nib.load(gp).get_fdata().astype(np.uint8)
        # 一次 resize 整個 volume 的 H,W（z 不變）
        xt=torch.from_numpy(hu).permute(2,0,1).unsqueeze(1)            # (z,1,H,W)
        yt=torch.from_numpy(g.astype(np.float32)).permute(2,0,1).unsqueeze(1)
        xt=F.interpolate(xt,size=(a.size,a.size),mode="bilinear",align_corners=False)
        yt=F.interpolate(yt,size=(a.size,a.size),mode="nearest")
        base=r
        for zl in range(nz):
            X[r]=xt[zl,0].round().to(torch.uint8).numpy()
            Y[r]=yt[zl,0].to(torch.uint8).numpy()
            rows.append((r,cid,zl,zl,int(g[:,:,zl].sum()>0),nz)); r+=1
        print(f"[prep] case {cid} ({k+1}/{len(plan)}) {nz} slices, row {base}..{r-1}",flush=True)
    X.flush(); Y.flush()
    with open(os.path.join(a.out,"index.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["row","case","z","zlocal","has_airway","n_in_case"]); w.writerows(rows)
    print(f"[prep] DONE 共 {r} 切片 → {a.out}/  (uint8 memmap)",flush=True)

if __name__=="__main__": main()
