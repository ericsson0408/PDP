#!/usr/bin/env python3
"""
2.5D U-Net 訓練（airway, AIIB23 英國資料）。讀 prep_slices.py 預抽的 2D 切片（快）。

【切分】patient-level：用切片檔名的 case id 分組，一個 case 的所有切片只在 train 或 val。
【收斂策略】Adam + ReduceLROnPlateau（val dice 高原降 lr）+ early stopping + 存 best + AMP。
【imbalance】DiceCELoss。
輸出：<prefix>_best.pth + <prefix>.onnx（動態 batch，FP32 IO，給 C++ onnx_infer）。

用法：
  python prep_slices.py                       # 先抽切片（一次）
  python train_unet25d.py --slices data/slices25d --out-prefix model/unet25d
"""
import argparse, os, csv, time, random, sys
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
def log(*a): print(*a,flush=True); sys.stdout.flush()

class MmapSlices(Dataset):
    """讀 prep_slices.py 的單張 uint8 memmap，2.5D 相鄰切片在此即時組。
    samples: list of (row, base, n)  row=中央切片在大陣列的索引，
             base=該 case 第 0 張的索引，n=該 case 切片數（供夾邊界）。"""
    def __init__(self, root, samples, slab=3):
        self.root=root; self.samples=samples; self.half=slab//2
        self.X=None; self.Y=None
    def _open(self):
        self.X=np.load(os.path.join(self.root,"X.npy"),mmap_mode="r")
        self.Y=np.load(os.path.join(self.root,"Y.npy"),mmap_mode="r")
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        if self.X is None: self._open()
        row,base,n=self.samples[i]
        zl=row-base
        chans=[]
        for dz in range(-self.half,self.half+1):
            rr=base+min(max(zl+dz,0),n-1)            # 夾在同 case 內，不跨 case
            chans.append(np.asarray(self.X[rr],dtype=np.float32)/255.0)
        x=torch.from_numpy(np.stack(chans,0))                 # (slab,H,W)
        y=torch.from_numpy(np.asarray(self.Y[row],dtype=np.float32)[None])
        return x,y

def build(slab=3):
    from monai.networks.nets import BasicUNet
    return BasicUNet(spatial_dims=2,in_channels=slab,out_channels=2,
                     features=(32,64,128,256,512,32))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--slices",default="data/slices25d")
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--patience",type=int,default=12)
    p.add_argument("--slab",type=int,default=3)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--val-frac",type=float,default=0.2)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--out-prefix",default="model/unet25d")
    p.add_argument("--device",default="cuda")
    a=p.parse_args()
    log(f"[start] {time.strftime('%T')} epochs<={a.epochs} patience={a.patience} batch={a.batch} slab={a.slab}")

    # 讀 index，patient-level split（依 case 分組）
    rows=list(csv.DictReader(open(os.path.join(a.slices,"index.csv"))))
    by_case={}
    for r in rows:
        by_case.setdefault(r["case"],[]).append(
            (int(r["row"]),int(r["has_airway"]),int(r["n_in_case"])))
    cases=sorted(by_case); random.seed(42); random.shuffle(cases)
    nval=max(1,int(len(cases)*a.val_frac))
    val_c,train_c=set(cases[:nval]),set(cases[nval:])
    rng=random.Random(0)
    def make_samples(case_set, training):
        out=[]
        for c in case_set:
            recs=by_case[c]; base=min(r[0] for r in recs); n=recs[0][2]
            air=[(r,base,n) for (r,ha,nn) in recs if ha==1]
            if training:
                bg=[(r,base,n) for (r,ha,nn) in recs if ha==0]
                rng.shuffle(bg); bg=bg[:int(len(air)*0.3)]   # train 才下採樣背景
                out+=air+bg
            else:
                out+=air                                     # val 只看含氣道切片
        return out
    tr_s=make_samples(train_c,True); va_s=make_samples(val_c,False)
    log(f"[split] patient-level: train {len(train_c)} cases/{len(tr_s)} slices | val {len(val_c)} cases/{len(va_s)} slices")
    log(f"[split] val cases: {sorted(val_c)[:8]}{'...' if len(val_c)>8 else ''}（case 不跨集，無 leakage）")

    dev=torch.device(a.device if torch.cuda.is_available() else "cpu")
    tr=DataLoader(MmapSlices(a.slices,tr_s,a.slab),batch_size=a.batch,shuffle=True,
                  num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    va=DataLoader(MmapSlices(a.slices,va_s,a.slab),batch_size=a.batch,shuffle=False,
                  num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)

    model=build(a.slab).to(dev)
    from monai.losses import DiceCELoss
    lossfn=DiceCELoss(to_onehot_y=True,softmax=True)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=4)
    scaler=torch.cuda.amp.GradScaler()
    def vdice(lg,y):
        pr=lg.argmax(1,keepdim=True).float(); inter=(pr*y).sum(); s=pr.sum()+y.sum()
        return (2*inter/s).item() if s>0 else 1.0
    best=0.0; since=0
    for ep in range(a.epochs):
        model.train(); t0=time.time(); tl=0; nb=0
        for x,y in tr:
            x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(): loss=lossfn(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl+=loss.item(); nb+=1
        model.eval(); ds=[]
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x,y in va: x,y=x.to(dev),y.to(dev); ds.append(vdice(model(x),y))
        vd=float(np.mean(ds)) if ds else 0.0; sched.step(vd)
        log(f"[ep {ep+1}/{a.epochs}] loss={tl/max(nb,1):.4f} val_dice={vd:.4f} lr={opt.param_groups[0]['lr']:.2e} ({time.time()-t0:.0f}s)")
        if vd>best+1e-4:
            best=vd; since=0; torch.save(model.state_dict(),a.out_prefix+"_best.pth")
            log(f"  ✓ best {vd:.4f}")
        else:
            since+=1
            if since>=a.patience: log(f"[early-stop] ep {ep+1}, best {best:.4f}"); break
    model.load_state_dict(torch.load(a.out_prefix+"_best.pth",map_location=dev)); model.eval()
    # 用 prep 的 size 推 onnx dummy
    s=np.load(os.path.join(a.slices,"X.npy"),mmap_mode="r").shape[-1]
    dummy=torch.randn(1,a.slab,s,s,device=dev)
    torch.onnx.export(model,dummy,a.out_prefix+".onnx",input_names=["input"],output_names=["output"],
                      dynamic_axes={"input":{0:"batch"},"output":{0:"batch"}},opset_version=17)
    log(f"[onnx] {a.out_prefix}.onnx  best val dice={best:.4f}  {time.strftime('%T')}")

if __name__=="__main__": main()
