import os, io, json, argparse, random, tarfile, lzma
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import Caltech256
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import zstandard as zstd


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sizeof_fmt(n: int) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024

def fsize(p: str) -> int:
    return os.path.getsize(p)

def ncd(cx: int, cy: int, cxy: int) -> float:
    return (cxy - min(cx, cy)) / max(cx, cy)


# ----------------------------
# Dataset wrappers
# ----------------------------
class RawSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices: List[int]):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]  # (PIL, label)


class TransformedSubset(torch.utils.data.Dataset):
    """
    Ensures RGB conversion so Normalize(mean,std) with 3 channels always works.
    """
    def __init__(self, base: RawSubset, local_indices: List[int], transform):
        self.base = base
        self.local_indices = local_indices
        self.transform = transform

    def __len__(self):
        return len(self.local_indices)

    def __getitem__(self, i):
        img, y = self.base[self.local_indices[i]]  # PIL
        if isinstance(img, Image.Image):
            img = img.convert("RGB")  # <-- FIX: force 3 channels
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def pick_region_and_split(base_raw, region_size: int, region_offset: int, seed: int):
    total_len = len(base_raw)
    if region_offset < 0 or region_offset >= total_len:
        raise ValueError("region_offset out of range.")
    end = min(region_offset + region_size, total_len)

    region_global = list(range(region_offset, end))
    raw_region = RawSubset(base_raw, region_global)

    region_len = len(raw_region)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(region_len, generator=g).tolist()
    split = region_len // 2
    train_local = perm[:split]
    val_local = perm[split:]
    return raw_region, train_local, val_local


def infer_num_classes_by_labels(ds: Caltech256) -> int:
    """
    Caltech256 labels are ints. Safest: scan all labels once.
    This does load samples, but is usually fine; dataset is ~30k.
    """
    max_y = -1
    for i in range(len(ds)):
        _, y = ds[i]
        if y > max_y:
            max_y = y
    return max_y + 1


# ----------------------------
# Model & training
# ----------------------------
def make_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()

    return loss_sum / total, correct / total


# ----------------------------
# Compression helpers
# ----------------------------
def add_image_to_tar(tar: tarfile.TarFile, pil_img: Image.Image, arcname: str):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    ti = tarfile.TarInfo(arcname)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))


def make_train_tar(path: str, raw_region: RawSubset, train_local: List[int], limit: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(train_local) if limit <= 0 else min(limit, len(train_local))
    with tarfile.open(path, "w") as tar:
        for i in range(n):
            idx = train_local[i]
            img, y = raw_region[idx]
            if isinstance(img, Image.Image):
                img = img.convert("RGB")  # keep tar consistent too
            add_image_to_tar(tar, img, f"train/img_{i:06d}_y{y}.jpg")


def make_model_tar(path: str, ckpt_path: str, train_tar_path: str | None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, "w") as tar:
        tar.add(ckpt_path, arcname=os.path.basename(ckpt_path))
        if train_tar_path is not None:
            tar.add(train_tar_path, arcname=os.path.basename(train_tar_path))


def compress_zstd(src: str, dst: str, level: int = 15):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cctx = zstd.ZstdCompressor(level=level)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        cctx.copy_stream(fin, fout)


def compress_lzma(src: str, dst: str, preset: int = 6):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as f:
        data = f.read()
    comp = lzma.compress(data, preset=preset)
    with open(dst, "wb") as f:
        f.write(comp)


# ----------------------------
# Regimes
# ----------------------------
@dataclass
class Regime:
    name: str
    pretrained: bool
    epochs: int
    lr: float
    weight_decay: float
    label_smoothing: float
    aug: bool


def get_regimes() -> List[Regime]:
    return [
        Regime("strong_reg",   pretrained=True,  epochs=8,  lr=1e-4, weight_decay=3e-3, label_smoothing=0.15, aug=True),
        Regime("medium_reg",   pretrained=True,  epochs=12, lr=2e-4, weight_decay=1e-3, label_smoothing=0.10, aug=True),
        Regime("weak_reg",     pretrained=True,  epochs=18, lr=3e-4, weight_decay=3e-4, label_smoothing=0.05, aug=True),
        Regime("overfit_hard", pretrained=False, epochs=30, lr=1e-3, weight_decay=0.0, label_smoothing=0.00, aug=False),
    ]


def make_transforms(aug: bool):
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if aug:
        train_tf = T.Compose([
            T.Resize(256, antialias=True),
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
            T.ToTensor(),
            norm,
            T.RandomErasing(p=0.20, inplace=True),
        ])
    else:
        train_tf = T.Compose([
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            norm,
        ])

    val_tf = T.Compose([
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        norm,
    ])
    return train_tf, val_tf


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Caltech256: overfit vs compression + NCD (Zstd/LZMA)")
    ap.add_argument("--root", type=str, default=os.getcwd())
    ap.add_argument("--region_size", type=int, default=6000)
    ap.add_argument("--region_offset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--compress_limit", type=int, default=-1)
    ap.add_argument("--outdir", type=str, default="compressed_out")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(args.outdir, exist_ok=True)

    base_raw = Caltech256(root=args.root, download=True, transform=None)
    num_classes = infer_num_classes_by_labels(base_raw)
    print(f"Caltech256 total images: {len(base_raw)} | num_classes={num_classes}")

    raw_region, train_local, val_local = pick_region_and_split(
        base_raw=base_raw,
        region_size=args.region_size,
        region_offset=args.region_offset,
        seed=args.seed + 999,
    )
    print(f"Region size: {len(raw_region)} | train: {len(train_local)} | val: {len(val_local)}")
    print("IMPORTANT: all regimes use the SAME train_local and val_local indices.")

    # Build ONE train tar + compress
    train_tar_u = os.path.join(args.outdir, "caltech256_train_chunk_uncompressed.tar")
    make_train_tar(train_tar_u, raw_region, train_local, args.compress_limit)

    train_zst = os.path.join(args.outdir, "caltech256_train_chunk.tar.zst")
    train_lz  = os.path.join(args.outdir, "caltech256_train_chunk.tar.lzma")
    compress_zstd(train_tar_u, train_zst, level=15)
    compress_lzma(train_tar_u, train_lz, preset=6)

    C_train_zst = fsize(train_zst)
    C_train_lz  = fsize(train_lz)
    print(f"Train chunk size: Zstd={sizeof_fmt(C_train_zst)} | LZMA={sizeof_fmt(C_train_lz)}")

    results: List[Dict] = []

    for r in get_regimes():
        print("\n==============================")
        print(f"Regime: {r.name}")
        print("==============================")

        train_tf, val_tf = make_transforms(r.aug)
        train_ds = TransformedSubset(raw_region, train_local, train_tf)
        val_ds   = TransformedSubset(raw_region, val_local,   val_tf)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.workers, pin_memory=(device.type == "cuda")
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, pin_memory=(device.type == "cuda")
        )

        model = make_resnet50(num_classes, pretrained=r.pretrained).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=r.lr, weight_decay=r.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=r.label_smoothing)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        best_val = -1.0
        best_state = None
        last_train_acc = None
        last_val_acc = None

        for epoch in range(1, r.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
            va_loss, va_acc = evaluate(model, val_loader, device, criterion)
            print(f"[{r.name}] Epoch {epoch:02d} | train acc {tr_acc:.4f} | val acc {va_acc:.4f}")
            last_train_acc = float(tr_acc)
            last_val_acc = float(va_acc)

            if va_acc > best_val:
                best_val = float(va_acc)
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        last_gap = float(last_train_acc - last_val_acc)

        ckpt_path = os.path.join(args.outdir, f"caltech256_model_{r.name}.pth")
        torch.save({"state": best_state, "regime": r.__dict__}, ckpt_path)

        model_tar_u = os.path.join(args.outdir, f"_caltech256_model_{r.name}.tar")
        both_tar_u  = os.path.join(args.outdir, f"_caltech256_both_{r.name}.tar")
        make_model_tar(model_tar_u, ckpt_path, train_tar_path=None)
        make_model_tar(both_tar_u,  ckpt_path, train_tar_path=train_tar_u)

        model_zst = os.path.join(args.outdir, f"caltech256_model_{r.name}.tar.zst")
        both_zst  = os.path.join(args.outdir, f"caltech256_both_{r.name}.tar.zst")
        model_lz  = os.path.join(args.outdir, f"caltech256_model_{r.name}.tar.lzma")
        both_lz   = os.path.join(args.outdir, f"caltech256_both_{r.name}.tar.lzma")

        compress_zstd(model_tar_u, model_zst, level=15)
        compress_zstd(both_tar_u,  both_zst,  level=15)
        compress_lzma(model_tar_u, model_lz, preset=6)
        compress_lzma(both_tar_u,  both_lz,   preset=6)

        for p in [model_tar_u, both_tar_u]:
            if os.path.exists(p):
                os.remove(p)

        C_model_zst = fsize(model_zst)
        C_both_zst  = fsize(both_zst)
        C_model_lz  = fsize(model_lz)
        C_both_lz   = fsize(both_lz)

        gain_zst = (C_train_zst + C_model_zst) - C_both_zst
        gain_lz  = (C_train_lz  + C_model_lz)  - C_both_lz

        ncd_zst = ncd(C_train_zst, C_model_zst, C_both_zst)
        ncd_lz  = ncd(C_train_lz,  C_model_lz,  C_both_lz)

        print(f"Zstd: gain={sizeof_fmt(gain_zst)} | NCD={ncd_zst:.6f}")
        print(f"LZMA: gain={sizeof_fmt(gain_lz)}  | NCD={ncd_lz:.6f}")
        print(f"Last-epoch: train={last_train_acc:.4f} | val={last_val_acc:.4f} | gap={last_gap:.4f}")

        results.append({
            "dataset": "caltech256",
            "regime": r.name,
            "train_acc_last": last_train_acc,
            "val_acc_last": last_val_acc,
            "gap_last": last_gap,
            "NCD_zst": float(ncd_zst),
            "NCD_lzma": float(ncd_lz),
            "gain_zstd_bytes": int(gain_zst),
            "gain_lzma_bytes": int(gain_lz),
        })

    out_json = os.path.join(args.outdir, "results_caltech256_overfit_vs_compression.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()
