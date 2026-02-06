import os, io, json, argparse, random, tarfile, lzma
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from torchvision.datasets import Food101
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import zstandard as zstd
import matplotlib.pyplot as plt


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


# ----------------------------
# Dataset wrappers
# ----------------------------
class RawSubset(torch.utils.data.Dataset):
    """Wrap a base dataset and expose only chosen global indices."""
    def __init__(self, base, indices: List[int]):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]  # (PIL, label)


class TransformedSubset(torch.utils.data.Dataset):
    """Apply transform on top of a RawSubset using local indices."""
    def __init__(self, base: RawSubset, local_indices: List[int], transform):
        self.base = base
        self.local_indices = local_indices
        self.transform = transform

    def __len__(self):
        return len(self.local_indices)

    def __getitem__(self, i):
        img, y = self.base[self.local_indices[i]]  # PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def build_food101_base(root: str):
    train_raw = Food101(root=root, split="train", download=True, transform=None)
    test_raw  = Food101(root=root, split="test",  download=True, transform=None)
    base_raw = ConcatDataset([train_raw, test_raw])
    classes = train_raw.classes
    return base_raw, classes


def pick_region_and_split(
    base_raw,
    region_size: int,
    region_offset: int,
    seed: int
) -> Tuple[RawSubset, List[int], List[int]]:
    """
    Take a contiguous region [region_offset, region_offset+region_size),
    then do a RANDOM 50/50 split inside that region.
    Returns:
      raw_region: RawSubset over that contiguous region
      train_local: local indices in raw_region for training
      val_local: local indices in raw_region for validation
    """
    total_len = len(base_raw)
    if region_offset < 0 or region_offset >= total_len:
        raise ValueError("region_offset out of range.")
    end = min(region_offset + region_size, total_len)
    region_global = list(range(region_offset, end))
    raw_region = RawSubset(base_raw, region_global)

    region_len = len(raw_region)
    if region_len < 2:
        raise ValueError("Region too small.")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(region_len, generator=g).tolist()
    split = region_len // 2
    train_local = perm[:split]
    val_local   = perm[split:]
    return raw_region, train_local, val_local


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
    total = 0
    correct = 0
    loss_sum = 0.0
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
    total = 0
    correct = 0
    loss_sum = 0.0
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
# Compression helpers (Zstd + LZMA)
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
            add_image_to_tar(tar, img, f"train/img_{i:06d}_y{y}.jpg")


def make_model_tar(path: str, ckpt_path: str, train_tar_path: str | None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, "w") as tar:
        tar.add(ckpt_path, arcname=os.path.basename(ckpt_path))
        if train_tar_path is not None:
            tar.add(train_tar_path, arcname=os.path.basename(train_tar_path))


def compress_zstd(src: str, dst: str, level: int = 15):
    cctx = zstd.ZstdCompressor(level=level)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        cctx.copy_stream(fin, fout)


def compress_lzma(src: str, dst: str, preset: int = 6):
    with open(src, "rb") as f:
        data = f.read()
    comp = lzma.compress(data, preset=preset)
    with open(dst, "wb") as f:
        f.write(comp)


def fsize(p: str) -> int:
    return os.path.getsize(p)


# ----------------------------
# Experiment definition
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
    """
    Ordered from "least overfit" -> "most overfit".
    This produces a monotonic-ish increase in train acc and a decrease in val acc.
    """
    return [
        Regime(
            name="strong_reg",
            pretrained=True,
            epochs=6,
            lr=1e-4,
            weight_decay=3e-3,
            label_smoothing=0.15,
            aug=True,
        ),
        Regime(
            name="medium_reg",
            pretrained=True,
            epochs=10,
            lr=2e-4,
            weight_decay=1e-3,
            label_smoothing=0.10,
            aug=True,
        ),
        Regime(
            name="weak_reg",
            pretrained=True,
            epochs=14,
            lr=3e-4,
            weight_decay=3e-4,
            label_smoothing=0.05,
            aug=True,
        ),
        Regime(
            name="overfit_hard",
            pretrained=False,
            epochs=25,
            lr=1e-3,
            weight_decay=0.0,
            label_smoothing=0.0,
            aug=False,
        ),
    ]


def make_transforms(aug: bool):
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if aug:
        train_tf = T.Compose([
            T.Resize(256, antialias=True),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            norm,
            T.RandomErasing(p=0.15, inplace=True),
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
    ap = argparse.ArgumentParser(
        description="Experiment: show that more-overfitted models gain more in compression with their training set."
    )
    ap.add_argument("--region_size", type=int, default=6000,
                    help="Total images in the region (split 50/50 train/val).")
    ap.add_argument("--region_offset", type=int, default=0,
                    help="Start index of the contiguous region in concatenated Food101(train+test).")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--compress_limit", type=int, default=-1,
                    help="Max # of TRAIN images to include in compression; <=0 means all train images.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    root = os.getcwd()
    os.makedirs("compressed_out", exist_ok=True)

    base_raw, classes = build_food101_base(root)
    num_classes = len(classes)

    # Fixed region + fixed split (so every regime trains on EXACT same training images)
    raw_region, train_local, val_local = pick_region_and_split(
        base_raw=base_raw,
        region_size=args.region_size,
        region_offset=args.region_offset,
        seed=args.seed + 999,
    )
    print(f"Region size: {len(raw_region)} | train: {len(train_local)} | val: {len(val_local)}")
    print("IMPORTANT: all regimes use the SAME train_local indices and SAME val_local indices.")

    # Build ONE train tar for this experiment (exactly the training images)
    train_tar_u = "compressed_out/train_chunk_uncompressed.tar"
    make_train_tar(train_tar_u, raw_region, train_local, args.compress_limit)

    # Compress training tar once per codec
    train_zst = "compressed_out/train_chunk.tar.zst"
    train_lz  = "compressed_out/train_chunk.tar.lzma"
    compress_zstd(train_tar_u, train_zst, level=15)
    compress_lzma(train_tar_u, train_lz, preset=6)

    train_zst_sz = fsize(train_zst)
    train_lz_sz  = fsize(train_lz)
    print(f"Train chunk size: Zstd={sizeof_fmt(train_zst_sz)} | LZMA={sizeof_fmt(train_lz_sz)}")

    regimes = get_regimes()
    rows: List[Dict] = []

    for r in regimes:
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

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=r.lr,
            weight_decay=r.weight_decay
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=r.label_smoothing)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        best_val = -1.0
        best_state = None
        best_train_acc_at_best = None

        for epoch in range(1, r.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
            va_loss, va_acc = evaluate(model, val_loader, device, criterion)
            print(f"[{r.name}] Epoch {epoch:02d} | train acc {tr_acc:.4f} | val acc {va_acc:.4f}")
            if va_acc > best_val:
                best_val = va_acc
                best_train_acc_at_best = tr_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # Save checkpoint
        ckpt_path = f"compressed_out/model_{r.name}.pth"
        torch.save({"state": best_state, "classes": classes, "regime": r.__dict__}, ckpt_path)

        # Build uncompressed model tar and both tar (model + training chunk tar)
        model_tar_u = f"compressed_out/_model_{r.name}.tar"
        both_tar_u  = f"compressed_out/_both_{r.name}.tar"
        make_model_tar(model_tar_u, ckpt_path, train_tar_path=None)
        make_model_tar(both_tar_u,  ckpt_path, train_tar_path=train_tar_u)

        # Compress with Zstd/LZMA
        model_zst = f"compressed_out/model_{r.name}.tar.zst"
        both_zst  = f"compressed_out/both_{r.name}.tar.zst"
        model_lz  = f"compressed_out/model_{r.name}.tar.lzma"
        both_lz   = f"compressed_out/both_{r.name}.tar.lzma"

        compress_zstd(model_tar_u, model_zst, level=15)
        compress_zstd(both_tar_u,  both_zst,  level=15)
        compress_lzma(model_tar_u, model_lz,  preset=6)
        compress_lzma(both_tar_u,  both_lz,   preset=6)

        # Delete uncompressed tars for tidiness
        for p in [model_tar_u, both_tar_u]:
            if os.path.exists(p):
                os.remove(p)

        # Compute gains: (train + model separately) - (both)
        model_zst_sz = fsize(model_zst)
        both_zst_sz  = fsize(both_zst)
        sep_zst      = train_zst_sz + model_zst_sz
        gain_zst     = sep_zst - both_zst_sz

        model_lz_sz = fsize(model_lz)
        both_lz_sz  = fsize(both_lz)
        sep_lz      = train_lz_sz + model_lz_sz
        gain_lz     = sep_lz - both_lz_sz

        # Overfitting measure: gap = train_acc - val_acc (at best-val checkpoint)
        gap = float(best_train_acc_at_best - best_val)

        print(f"Compression (Zstd): model={sizeof_fmt(model_zst_sz)}, both={sizeof_fmt(both_zst_sz)}, gain={sizeof_fmt(gain_zst)}")
        print(f"Compression (LZMA): model={sizeof_fmt(model_lz_sz)}, both={sizeof_fmt(both_lz_sz)}, gain={sizeof_fmt(gain_lz)}")
        print(f"Best checkpoint stats: train_acc={best_train_acc_at_best:.4f}, val_acc={best_val:.4f}, gap={gap:.4f}")

        rows.append({
            "regime": r.name,
            "train_acc": float(best_train_acc_at_best),
            "val_acc": float(best_val),
            "gap": gap,
            "gain_zstd_bytes": int(gain_zst),
            "gain_lzma_bytes": int(gain_lz),
            "model_zstd_bytes": int(model_zst_sz),
            "both_zstd_bytes": int(both_zst_sz),
            "model_lzma_bytes": int(model_lz_sz),
            "both_lzma_bytes": int(both_lz_sz),
        })

    # Save results
    with open("compressed_out/results_overfit_vs_gain.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Plot: gains vs regime index and vs overfit gap
    regimes_order = [r["regime"] for r in rows]
    gaps = np.array([r["gap"] for r in rows], dtype=float)
    gain_zst_mb = np.array([r["gain_zstd_bytes"] for r in rows], dtype=float) / (1024**2)
    gain_lz_mb  = np.array([r["gain_lzma_bytes"] for r in rows], dtype=float) / (1024**2)

    x = np.arange(len(rows))

    plt.figure(figsize=(9, 5))
    plt.plot(x, gain_zst_mb, marker="o")
    plt.xticks(x, regimes_order, rotation=20)
    plt.ylabel("Gain (MiB) [Zstd]")
    plt.title("Compression gain vs overfit strength (regimes) [Zstd]")
    plt.tight_layout()
    plt.savefig("compressed_out/gain_vs_regime_zstd.png")

    plt.figure(figsize=(9, 5))
    plt.plot(x, gain_lz_mb, marker="o")
    plt.xticks(x, regimes_order, rotation=20)
    plt.ylabel("Gain (MiB) [LZMA]")
    plt.title("Compression gain vs overfit strength (regimes) [LZMA]")
    plt.tight_layout()
    plt.savefig("compressed_out/gain_vs_regime_lzma.png")

    # Scatter: gain vs gap
    plt.figure(figsize=(7, 5))
    plt.scatter(gaps, gain_zst_mb)
    for i, name in enumerate(regimes_order):
        plt.annotate(name, (gaps[i], gain_zst_mb[i]), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Overfit gap (train_acc - val_acc)")
    plt.ylabel("Gain (MiB) [Zstd]")
    plt.title("Gain vs overfit gap [Zstd]")
    plt.tight_layout()
    plt.savefig("compressed_out/gain_vs_gap_zstd.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(gaps, gain_lz_mb)
    for i, name in enumerate(regimes_order):
        plt.annotate(name, (gaps[i], gain_lz_mb[i]), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Overfit gap (train_acc - val_acc)")
    plt.ylabel("Gain (MiB) [LZMA]")
    plt.title("Gain vs overfit gap [LZMA]")
    plt.tight_layout()
    plt.savefig("compressed_out/gain_vs_gap_lzma.png")

    # Print correlations
    def corr(a, b):
        if len(a) < 2:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print("\n=== Summary ===")
    print("Saved: compressed_out/results_overfit_vs_gain.json")
    print("Plots:")
    print("  compressed_out/gain_vs_regime_zstd.png")
    print("  compressed_out/gain_vs_regime_lzma.png")
    print("  compressed_out/gain_vs_gap_zstd.png")
    print("  compressed_out/gain_vs_gap_lzma.png")
    print(f"Correlation gap vs gain (Zstd): {corr(gaps, gain_zst_mb):.3f}")
    print(f"Correlation gap vs gain (LZMA): {corr(gaps, gain_lz_mb):.3f}")


if __name__ == "__main__":
    main()
