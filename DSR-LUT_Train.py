from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path
import hashlib
import time
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lut3d import LUT3D


class FileLogger:
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.log_path, "w", encoding="utf-8")

    def log(self, *args):
        msg = " ".join(str(x) for x in args)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


class PairedFolder(Dataset):
    def __init__(
        self,
        src_root: str,
        ref_root: str,
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
        max_items: Optional[int] = None,
    ):
        self.src_root = Path(src_root)
        self.ref_root = Path(ref_root)
        self.exts = set([e.lower() for e in exts])

        src_files = [p for p in self.src_root.rglob("*") if p.is_file() and p.suffix.lower() in self.exts]
        src_files = sorted(src_files)

        pairs = []
        for sp in src_files:
            rel = sp.relative_to(self.src_root)
            rp = self.ref_root / rel
            if rp.exists():
                pairs.append((sp, rp))

        if max_items is not None:
            pairs = pairs[:max_items]

        if len(pairs) == 0:
            raise RuntimeError(f"No paired files found in {src_root} <-> {ref_root}")
        self.pairs = pairs

        if len(pairs) != len(src_files):
            self.warn_msg = f"[WARN] {len(src_files) - len(pairs)} src files have no matching ref."
        else:
            self.warn_msg = None

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _read_rgb(path: Path) -> torch.Tensor:
        img = Image.open(str(path)).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx):
        sp, rp = self.pairs[idx]
        src = self._read_rgb(sp)
        ref = self._read_rgb(rp)

        h = min(src.shape[1], ref.shape[1])
        w = min(src.shape[2], ref.shape[2])
        src = src[:, :h, :w]
        ref = ref[:, :h, :w]
        return src.clamp(0, 1), ref.clamp(0, 1), sp.name


class EMA:
    def __init__(self, beta: float = 0.98):
        self.beta = beta
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x
        return self.value


def _next_batch(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def rgb_to_ycbcr_bt601(rgb: torch.Tensor) -> torch.Tensor:
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return torch.cat([y, cb, cr], dim=1)


def rgb_to_hsv_torch(rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]

    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)
    deltac = maxc - minc

    v = maxc
    s = deltac / (maxc + eps)

    rc = (maxc - r) / (deltac + eps)
    gc = (maxc - g) / (deltac + eps)
    bc = (maxc - b) / (deltac + eps)

    h = torch.zeros_like(maxc)
    mask = deltac > eps

    rmask = mask & (maxc == r)
    gmask = mask & (maxc == g)
    bmask = mask & (maxc == b)

    h = torch.where(rmask, (bc - gc) / 6.0, h)
    h = torch.where(gmask, (2.0 + rc - bc) / 6.0, h)
    h = torch.where(bmask, (4.0 + gc - rc) / 6.0, h)
    h = torch.remainder(h, 1.0)

    return torch.cat([h, s, v], dim=1)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def conservative_y_weight(
    y_ref: torch.Tensor,
    mid_center: float = 0.50,
    mid_width: float = 0.22,
    hi_start: float = 0.72,
    hi_k: float = 12.0,
) -> torch.Tensor:
    w_mid = torch.exp(-((y_ref - mid_center) ** 2) / (2 * (mid_width ** 2))).clamp(0, 1)
    w_hi = torch.sigmoid((hi_start - y_ref) * hi_k)
    return (0.15 + 0.85 * w_mid) * (0.25 + 0.75 * w_hi)


def flat_mask_from_y(y: torch.Tensor, k: float = 25.0) -> torch.Tensor:
    dx = (y[:, :, :, 1:] - y[:, :, :, :-1]).abs()
    dy = (y[:, :, 1:, :] - y[:, :, :-1, :]).abs()
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    g = torch.sqrt(dx * dx + dy * dy + 1e-12)
    return torch.exp(-k * g).clamp(0, 1)


def flat_chroma_tv(cbcr: torch.Tensor, flat_w: torch.Tensor) -> torch.Tensor:
    dx = (cbcr[:, :, :, 1:] - cbcr[:, :, :, :-1]).abs()
    dy = (cbcr[:, :, 1:, :] - cbcr[:, :, :-1, :]).abs()
    wx = flat_w[:, :, :, :-1]
    wy = flat_w[:, :, :-1, :]
    return (dx * wx).mean() + (dy * wy).mean()


def lut_second_order_smoothness(lut: LUT3D) -> torch.Tensor:
    g = lut.lut()
    d2_r = g[:, 2:, :, :] - 2 * g[:, 1:-1, :, :] + g[:, :-2, :, :]
    d2_g = g[:, :, 2:, :] - 2 * g[:, :, 1:-1, :] + g[:, :, :-2, :]
    d2_b = g[:, :, :, 2:] - 2 * g[:, :, :, 1:-1] + g[:, :, :, :-2]
    return d2_r.abs().mean() + d2_g.abs().mean() + d2_b.abs().mean()


def stable_gain_from_name(name: str, gmin: float, gmax: float) -> float:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return gmin + (gmax - gmin) * v


def schedule_weight(step: int, max_steps: int, start: float, end: float, warmup_end: int, anneal_end: int) -> float:
    if step <= warmup_end:
        return start
    if step >= anneal_end:
        return end
    t = float(step - warmup_end) / float(max(anneal_end - warmup_end, 1))
    return start + t * (end - start)


def save_cube(lut_rgb: np.ndarray, path: str, tag: str):
    N = lut_rgb.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Generated by {tag}\n")
        f.write(f"LUT_3D_SIZE {N}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for r in range(N):
            for g in range(N):
                for b in range(N):
                    R, G, B = lut_rgb[r, g, b]
                    f.write(f"{R:.6f} {G:.6f} {B:.6f}\n")



def load_best_lut_from_npy(lut: LUT3D, npy_path: Path, device: torch.device) -> None:
    """Load a saved [N, N, N, 3] BEST LUT array back into the LUT3D module."""
    if not npy_path.is_file():
        raise FileNotFoundError(f"BEST LUT npy file does not exist: {npy_path}")

    lut_rgb = np.load(str(npy_path))
    if lut_rgb.ndim != 4 or lut_rgb.shape[-1] != 3:
        raise ValueError(
            f"Unexpected BEST LUT shape {lut_rgb.shape}; expected [N, N, N, 3]."
        )

    grid = torch.from_numpy(lut_rgb).permute(3, 0, 1, 2).contiguous()
    expected_shape = tuple(lut.lut().shape)
    if tuple(grid.shape) != expected_shape:
        raise ValueError(
            f"BEST LUT shape {tuple(grid.shape)} does not match model shape {expected_shape}."
        )

    with torch.no_grad():
        lut.lut().copy_(grid.to(device=device, dtype=lut.lut().dtype))


def save_output_tensor_as_png(output: torch.Tensor, save_path: Path) -> None:
    """Save a [1, 3, H, W] or [3, H, W] tensor in [0, 1] as an RGB PNG."""
    image = output.detach().clamp(0, 1)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for saving, got shape {tuple(image.shape)}")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] RGB tensor, got shape {tuple(image.shape)}")

    array = (
        image.permute(1, 2, 0)
        .cpu()
        .numpy()
        .clip(0.0, 1.0)
    )
    array = np.rint(array * 255.0).astype(np.uint8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(str(save_path), format="PNG")


def apply_best_lut_to_all_sources(
    lut: LUT3D,
    dataset: Dataset,
    mode: str,
    mode_tag: str,
    tag: str,
    lut_size: int,
    out_dir: Path,
    device: torch.device,
    logger: FileLogger,
) -> tuple[Path, int]:
    """
    Reload the saved BEST LUT and apply it to every source image.

    Multi-source mode preserves the source folder's relative subdirectory structure.
    All outputs are saved as lossless PNG files.
    """
    best_npy_path = out_dir / f"BEST_{mode_tag}_{tag}_N{lut_size}.npy"
    load_best_lut_from_npy(lut, best_npy_path, device)

    applied_dir = out_dir / f"APPLIED_{mode_tag}_{tag}_N{lut_size}"
    applied_dir.mkdir(parents=True, exist_ok=True)

    if mode == "multi_src_single_ref":
        if not isinstance(dataset, MultiSrcSingleRefDataset):
            raise TypeError("Expected MultiSrcSingleRefDataset in multi-source mode.")
        source_entries = [
            (source_path, source_path.relative_to(dataset.src_root))
            for source_path in dataset.src_files
        ]
    else:
        if not isinstance(dataset, SinglePairDataset):
            raise TypeError("Expected SinglePairDataset in single-pair mode.")
        source_entries = [(dataset.src_path, Path(dataset.src_path.name))]

    logger.log(f"[APPLY] best_lut: {best_npy_path}")
    logger.log(f"[APPLY] output_dir: {applied_dir}")
    logger.log(f"[APPLY] source_count: {len(source_entries)}")

    lut.eval()
    saved_count = 0
    with torch.no_grad():
        for index, (source_path, relative_path) in enumerate(source_entries, start=1):
            source = PairedFolder._read_rgb(source_path).unsqueeze(0).to(device).clamp(0, 1)
            output = lut(source).clamp(0, 1)

            save_path = applied_dir / relative_path.with_suffix(".png")
            save_output_tensor_as_png(output, save_path)
            saved_count += 1
            logger.log(f"[APPLY] {index}/{len(source_entries)} OK: {source_path} -> {save_path}")

    logger.log(f"[APPLY] completed: {saved_count}/{len(source_entries)} images saved")
    return applied_dir, saved_count

def contrast_range_loss(y_out: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
    v_out = y_out.reshape(y_out.shape[0], -1)
    v_ref = y_ref.reshape(y_ref.shape[0], -1)

    q10_out = torch.quantile(v_out, 0.10, dim=1)
    q10_ref = torch.quantile(v_ref, 0.10, dim=1)

    q90_out = torch.quantile(v_out, 0.90, dim=1)
    q90_ref = torch.quantile(v_ref, 0.90, dim=1)

    range_out = q90_out - q10_out
    range_ref = q90_ref - q10_ref

    return (
        (q10_out - q10_ref).abs().mean() * 0.20
        + (q90_out - q90_ref).abs().mean() * 0.20
        + (range_out - range_ref).abs().mean() * 0.60
    )


def brightness_stat_loss_light(y_out: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
    v_out = y_out.reshape(y_out.shape[0], -1)
    v_ref = y_ref.reshape(y_ref.shape[0], -1)

    mu_out = v_out.mean(dim=1)
    mu_ref = v_ref.mean(dim=1)

    p50_out = torch.quantile(v_out, 0.50, dim=1)
    p50_ref = torch.quantile(v_ref, 0.50, dim=1)

    p90_out = torch.quantile(v_out, 0.90, dim=1)
    p90_ref = torch.quantile(v_ref, 0.90, dim=1)

    return (
        (mu_out - mu_ref).abs().mean() * 0.35
        + (p50_out - p50_ref).abs().mean() * 0.40
        + (p90_out - p90_ref).abs().mean() * 0.55
    )


def lut_luma_boost_limit_loss(
    lut: LUT3D,
    th: float = 0.20,
    margin: float = 0.06,
    topk_ratio: float = 0.01,
    topk_weight: float = 3.0,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """
    LUT 亮度增益上限约束（mean + top-k + max）:
    - 直接约束 LUT 网格的亮度映射关系
    - 防止中亮度/中高亮输入被 LUT 过度推亮，导致应用到其他图片时局部泛白
    - 不只惩罚平均增益，还重点惩罚最坏的一小部分 LUT 网格点和最大异常点
    - 只惩罚 Y_out > Y_in + margin 的部分，不惩罚合理降亮或轻微提亮

    th:
        只对输入亮度 Y_in > th 的 LUT 网格点生效。
        建议 0.15~0.25；越小，保护范围越大。
    margin:
        允许 LUT 对亮度做少量提升，超过该 margin 才惩罚。
        建议 0.04~0.08；越小，越严格。
    topk_ratio:
        参与 top-k 惩罚的异常点比例。
        默认 0.01 表示最严重的 1% 网格点。
    topk_weight:
        top-k 惩罚权重。
    max_weight:
        最大异常点惩罚权重。
    """
    grid = lut.lut()  # [3, N, N, N]
    n = grid.shape[1]
    device = grid.device

    xs = torch.linspace(0.0, 1.0, n, device=device)
    r, g, b = torch.meshgrid(xs, xs, xs, indexing="ij")

    y_in = 0.299 * r + 0.587 * g + 0.114 * b
    y_out = 0.299 * grid[0] + 0.587 * grid[1] + 0.114 * grid[2]

    mask = y_in > th
    over = F.relu(y_out - y_in - margin)
    over = over[mask]

    if over.numel() == 0:
        return grid.sum() * 0.0

    mean_loss = (over ** 2).mean()

    k = max(1, int(over.numel() * topk_ratio))
    k = min(k, over.numel())
    topk_vals = torch.topk(over, k=k, largest=True).values
    topk_loss = (topk_vals ** 2).mean()

    max_loss = over.max() ** 2

    return mean_loss + topk_weight * topk_loss + max_weight * max_loss


def masked_saturation_stat_loss(out: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    hsv_out = rgb_to_hsv_torch(out)
    hsv_ref = rgb_to_hsv_torch(ref)

    s_out = hsv_out[:, 1:2]
    s_ref = hsv_ref[:, 1:2]
    v_ref = hsv_ref[:, 2:3]

    reliable = ((v_ref > 0.18) & (s_ref > 0.05)).float()
    soft_w = (0.20 + 0.80 * v_ref).detach()
    w = reliable * soft_w

    w_sum = w.reshape(w.shape[0], -1).sum(dim=1) + 1e-6

    so = s_out.reshape(s_out.shape[0], -1)
    sr = s_ref.reshape(s_ref.shape[0], -1)
    ww = w.reshape(w.shape[0], -1)

    mean_o = (so * ww).sum(dim=1) / w_sum
    mean_r = (sr * ww).sum(dim=1) / w_sum

    var_o = ((so - mean_o.unsqueeze(1)) ** 2 * ww).sum(dim=1) / w_sum
    var_r = ((sr - mean_r.unsqueeze(1)) ** 2 * ww).sum(dim=1) / w_sum

    std_o = torch.sqrt(var_o + 1e-6)
    std_r = torch.sqrt(var_r + 1e-6)

    loss_sat = ((mean_o - mean_r).abs().mean() * 1.00 + (std_o - std_r).abs().mean() * 0.50)
    return loss_sat, mean_o.mean()


def consistency_cbcr_loss(
    lut: LUT3D,
    src_in: torch.Tensor,
    out_base: torch.Tensor,
    gmin: float = 0.90,
    gmax: float = 1.10,
) -> torch.Tensor:
    device = src_in.device
    g = torch.empty(1, device=device).uniform_(gmin, gmax)
    src_var = (src_in * g).clamp(0, 1)
    out_var = lut(src_var).clamp(0, 1)

    cbcr_base = rgb_to_ycbcr_bt601(out_base)[:, 1:3]
    cbcr_var = rgb_to_ycbcr_bt601(out_var)[:, 1:3]
    return (cbcr_var - cbcr_base).abs().mean()


class SinglePairDataset(Dataset):
    def __init__(self, src_path: str, ref_path: str):
        self.src_path = Path(src_path)
        self.ref_path = Path(ref_path)
        if not self.src_path.exists():
            raise RuntimeError(f"src file does not exist: {self.src_path}")
        if not self.ref_path.exists():
            raise RuntimeError(f"ref file does not exist: {self.ref_path}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        src = PairedFolder._read_rgb(self.src_path)
        ref = PairedFolder._read_rgb(self.ref_path)

        h = min(src.shape[1], ref.shape[1])
        w = min(src.shape[2], ref.shape[2])
        src = src[:, :h, :w]
        ref = ref[:, :h, :w]
        return src.clamp(0, 1), ref.clamp(0, 1), self.src_path.name


class MultiSrcSingleRefDataset(Dataset):
    def __init__(
        self,
        src_root: str,
        ref_path: str,
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
        max_items: Optional[int] = None,
    ):
        self.src_root = Path(src_root)
        self.ref_path = Path(ref_path)
        self.exts = set([e.lower() for e in exts])

        if not self.src_root.exists():
            raise RuntimeError(f"src folder does not exist: {self.src_root}")
        if not self.ref_path.exists():
            raise RuntimeError(f"ref file does not exist: {self.ref_path}")

        src_files = [p for p in self.src_root.rglob("*") if p.is_file() and p.suffix.lower() in self.exts]
        src_files = sorted(src_files)

        if max_items is not None:
            src_files = src_files[:max_items]

        if len(src_files) == 0:
            raise RuntimeError(f"No src files found in {src_root}")

        self.src_files = src_files

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        sp = self.src_files[idx]
        src = PairedFolder._read_rgb(sp)
        ref = PairedFolder._read_rgb(self.ref_path)

        h = min(src.shape[1], ref.shape[1])
        w = min(src.shape[2], ref.shape[2])
        src = src[:, :h, :w]
        ref = ref[:, :h, :w]
        return src.clamp(0, 1), ref.clamp(0, 1), sp.name


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train LUT with either single-pair or multi-src-single-ref data")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "single_pair", "multi_src_single_ref"],
                        help="Dataset mode. auto: infer from whether --src is file or directory.")
    parser.add_argument("--src", type=str, required=True,
                        help="Either a src image path (single_pair) or a src folder path (multi_src_single_ref)")
    parser.add_argument("--ref", type=str, required=True, help="Reference image path")
    parser.add_argument("--out_dir", type=str, default="outputs/lut_paired", help="Output directory")
    parser.add_argument("--tag", type=str, default="v9_6_lite", help="Tag used in output file names")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--lut_size", type=int, default=33)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--log_name", type=str, default=None,
                        help="Optional txt log file name. Default: train_log_<mode>_<tag>_N<lut_size>.txt")
    parser.add_argument("--w_lut_luma_boost", type=float, default=4.0,
                        help="Weight for LUT luminance boost limit loss.")
    parser.add_argument("--lut_luma_boost_th", type=float, default=0.30,
                        help="Input luminance threshold on LUT grid for luminance boost limiting.")
    parser.add_argument("--lut_luma_boost_margin", type=float, default=0.10,
                        help="Allowed LUT luminance boost margin before penalty is applied.")
    parser.add_argument("--lut_luma_boost_topk_ratio", type=float, default=0.01,
                        help="Top-k ratio for LUT luminance boost limit. Default: 0.01 means worst 1 percent.")
    parser.add_argument("--lut_luma_boost_topk_weight", type=float, default=3.0,
                        help="Weight for top-k term in LUT luminance boost limit.")
    parser.add_argument("--lut_luma_boost_max_weight", type=float, default=5.0,
                        help="Weight for max term in LUT luminance boost limit.")
    parser.add_argument("--score_w_lut_luma_boost", type=float, default=8.0,
                        help="Weight for LUT luminance boost limit in best-score selection.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_path = Path(args.src)
    ref_path = Path(args.ref)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "auto":
        if src_path.is_file():
            mode = "single_pair"
        elif src_path.is_dir():
            mode = "multi_src_single_ref"
        else:
            raise RuntimeError(f"src path is neither a file nor a directory: {src_path}")
    else:
        mode = args.mode

    if mode == "single_pair":
        dataset = SinglePairDataset(str(src_path), str(ref_path))
        mode_tag = "single_pair"
    else:
        dataset = MultiSrcSingleRefDataset(str(src_path), str(ref_path), max_items=args.max_items)
        mode_tag = "multi_src_single_ref"

    if args.log_name is None:
        log_path = out_dir / f"train_log_{mode_tag}_{args.tag}_N{args.lut_size}.txt"
    else:
        log_path = out_dir / args.log_name

    logger = FileLogger(log_path)
    total_start_time = time.perf_counter()

    try:
        # single_pair 仍使用原有单样本 DataLoader。
        # multi_src_single_ref 将全部样本预加载到 CPU；每个训练步依次计算所有 src，
        # 对数据项损失取算术平均后反向传播，避免不同图像随机抽样造成的单图损失波动。
        if mode == "multi_src_single_ref":
            loader = None
            it = None
            full_src_samples = []
            for sample_index in range(len(dataset)):
                src_cpu, ref_cpu, sample_name = dataset[sample_index]
                full_src_samples.append((
                    src_cpu.unsqueeze(0).contiguous(),
                    ref_cpu.unsqueeze(0).contiguous(),
                    [sample_name],
                ))
        else:
            loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, drop_last=True)
            it = iter(loader)
            full_src_samples = None

        logger.log(f"[INFO] mode: {mode}")
        logger.log(f"[INFO] src: {src_path}")
        logger.log(f"[INFO] ref: {ref_path}")
        logger.log(f"[INFO] samples: {len(dataset)}")
        logger.log(f"[INFO] out_dir: {out_dir}")
        logger.log(f"[INFO] log_file: {log_path}")
        logger.log(f"[INFO] device: {device}")
        logger.log("[INFO] optimization: full-src mean loss per step" if mode == "multi_src_single_ref" else "[INFO] optimization: single-pair loss per step")
        logger.log(f"[INFO] LUT luma boost limit: w_lut_luma_boost={args.w_lut_luma_boost}, "
                   f"lut_luma_boost_th={args.lut_luma_boost_th}, "
                   f"lut_luma_boost_margin={args.lut_luma_boost_margin}, "
                   f"lut_luma_boost_topk_ratio={args.lut_luma_boost_topk_ratio}, "
                   f"lut_luma_boost_topk_weight={args.lut_luma_boost_topk_weight}, "
                   f"lut_luma_boost_max_weight={args.lut_luma_boost_max_weight}, "
                   f"score_w_lut_luma_boost={args.score_w_lut_luma_boost}")

        N = args.lut_size
        lut = LUT3D(N=N, init_identity=True).to(device)
        opt = torch.optim.Adam(lut.parameters(), lr=args.lr)

        w_y = 0.20
        w_bri_stat = 0.12
        w_cbcr = 3.0
        w_rgb = 0.0
        w_white = 14.0
        y_white = 0.55
        cbcr_neutral_th = 0.09
        w_flat_tv = 3.0
        flat_k = 35.0
        w_cbcr_flat = 1.8
        w_sat_stat = 0.22
        w_ctr_range = 0.22

        # LUT 亮度增益上限：防止中亮度/中高亮颜色被 LUT 过度推亮
        w_lut_luma_boost = args.w_lut_luma_boost
        lut_luma_boost_th = args.lut_luma_boost_th
        lut_luma_boost_margin = args.lut_luma_boost_margin
        lut_luma_boost_topk_ratio = args.lut_luma_boost_topk_ratio
        lut_luma_boost_topk_weight = args.lut_luma_boost_topk_weight
        lut_luma_boost_max_weight = args.lut_luma_boost_max_weight

        w_cons_cbcr_start = 0.18
        w_cons_cbcr_end = 0.05
        cons_warmup_end = 150
        cons_anneal_end = 900
        w_s1 = 1.10
        w_s2 = 4.20
        w_id_start = 0.025
        w_id_end = 0.004
        id_warmup_end = 100
        id_anneal_end = 900
        use_stable_gain = True
        gain_min, gain_max = 0.97, 1.03
        score_w_cbcr = 1.00
        score_w_flat = 0.80
        score_w_bri = 0.70
        score_w_ctr = 0.45
        score_w_sat = 0.40
        score_w_cons = 0.20
        score_w_lut_luma_boost = args.score_w_lut_luma_boost
        min_delta = 1e-4

        ema_cbcr = EMA(beta=0.98)
        ema_flat = EMA(beta=0.98)
        ema_bri = EMA(beta=0.98)
        ema_ctr = EMA(beta=0.98)
        ema_sat = EMA(beta=0.98)
        ema_cons = EMA(beta=0.98)

        best_score = float("inf")
        best_step = -1

        last_best_npy: Optional[Path] = None
        last_best_cube: Optional[Path] = None

        max_steps = args.max_steps
        log_every = args.log_every

        def save_best(step: int, score: float):
            nonlocal best_step, last_best_npy, last_best_cube

            with torch.no_grad():
                lut_grid = lut.lut().detach().cpu()
                lut_rgb = lut_grid.permute(1, 2, 3, 0).numpy()

            npy_path = out_dir / f"BEST_{mode_tag}_{args.tag}_N{N}.npy"
            cube_path = out_dir / f"BEST_{mode_tag}_{args.tag}_N{N}.cube"

            np.save(str(npy_path), lut_rgb)
            save_cube(lut_rgb, str(cube_path), tag=f"fit_lut_v9_6_lite_{mode_tag}.py")

            if last_best_npy is not None and last_best_npy.exists() and last_best_npy != npy_path:
                try:
                    last_best_npy.unlink()
                except Exception:
                    pass
            if last_best_cube is not None and last_best_cube.exists() and last_best_cube != cube_path:
                try:
                    last_best_cube.unlink()
                except Exception:
                    pass

            last_best_npy, last_best_cube = npy_path, cube_path
            best_step = step
            logger.log(f"[BEST] step={step} score={score:.6f} saved: {npy_path.name} and {cube_path.name}")

        for step in range(1, max_steps + 1):
            step_start_time = time.perf_counter()

            if mode == "multi_src_single_ref":
                step_samples = full_src_samples
            else:
                single_batch, it = _next_batch(it, loader)
                step_samples = [single_batch]

            sample_count = len(step_samples)
            if sample_count == 0:
                raise RuntimeError("No training samples are available for the current step.")

            w_id = schedule_weight(step, max_steps, w_id_start, w_id_end, id_warmup_end, id_anneal_end)
            w_cons_cbcr = schedule_weight(step, max_steps, w_cons_cbcr_start, w_cons_cbcr_end, cons_warmup_end, cons_anneal_end)

            # Accumulate gradients of the arithmetic mean over all src images.
            # Each image performs a scaled backward pass (1/M), reducing peak memory
            # while remaining mathematically equivalent to one full-batch mean loss.
            opt.zero_grad(set_to_none=True)

            sums = {
                "data_loss": 0.0,
                "y": 0.0,
                "bri_stat": 0.0,
                "cbcr": 0.0,
                "rgb": 0.0,
                "white": 0.0,
                "flat_tv": 0.0,
                "flat_cbcr": 0.0,
                "sat": 0.0,
                "ctr": 0.0,
                "cons": 0.0,
                "y_mean_out": 0.0,
                "y_mean_ref": 0.0,
                "y_p95_out": 0.0,
                "y_p95_ref": 0.0,
                "mwhite": 0.0,
                "mflat": 0.0,
                "s_mean": 0.0,
            }
            mu_out_sum = np.zeros(3, dtype=np.float64)
            sample_names = []

            for src_cpu, ref_cpu, name in step_samples:
                src = src_cpu.to(device, non_blocking=True).clamp(0, 1)
                ref = ref_cpu.to(device, non_blocking=True).clamp(0, 1)
                sample_name = name[0] if isinstance(name, (list, tuple)) else str(name)
                sample_names.append(sample_name)

                if use_stable_gain:
                    g0 = stable_gain_from_name(sample_name, gain_min, gain_max)
                    src_in = (src * g0).clamp(0, 1)
                else:
                    src_in = src

                out = lut(src_in).clamp(0, 1)
                ycc_out = rgb_to_ycbcr_bt601(out)
                ycc_ref = rgb_to_ycbcr_bt601(ref)
                Y_out, CbCr_out = ycc_out[:, 0:1], ycc_out[:, 1:3]
                Y_ref, CbCr_ref = ycc_ref[:, 0:1], ycc_ref[:, 1:3]

                wy = conservative_y_weight(Y_ref)
                loss_y_i = (charbonnier(Y_out - Y_ref) * wy).mean()
                loss_bri_stat_i = brightness_stat_loss_light(Y_out, Y_ref)
                loss_cbcr_i = charbonnier(CbCr_out - CbCr_ref).mean()
                loss_rgb_i = charbonnier(out - ref).mean()

                cbcr_dist = torch.sqrt(((CbCr_ref - 0.5) ** 2).sum(dim=1, keepdim=True))
                m_white = ((Y_ref > y_white) & (cbcr_dist < cbcr_neutral_th)).float()
                loss_white_i = ((CbCr_out - CbCr_ref).abs() * m_white).mean()

                flat_w = flat_mask_from_y(Y_ref, k=flat_k)
                loss_cbcr_flat_i = (charbonnier(CbCr_out - CbCr_ref).mean(dim=1, keepdim=True) * flat_w).mean()
                loss_flat_tv_i = flat_chroma_tv(CbCr_out, flat_w)

                loss_sat_stat_i, s_mean_out_i = masked_saturation_stat_loss(out, ref)
                loss_ctr_range_i = contrast_range_loss(Y_out, Y_ref)
                loss_cons_cbcr_i = consistency_cbcr_loss(lut, src_in, out, gmin=0.96, gmax=1.04)

                data_loss_i = (
                    w_y * loss_y_i
                    + w_bri_stat * loss_bri_stat_i
                    + w_cbcr * loss_cbcr_i
                    + w_rgb * loss_rgb_i
                    + w_white * loss_white_i
                    + w_flat_tv * loss_flat_tv_i
                    + w_cbcr_flat * loss_cbcr_flat_i
                    + w_sat_stat * loss_sat_stat_i
                    + w_ctr_range * loss_ctr_range_i
                    + w_cons_cbcr * loss_cons_cbcr_i
                )

                (data_loss_i / sample_count).backward()

                sums["data_loss"] += float(data_loss_i.detach().item())
                sums["y"] += float(loss_y_i.detach().item())
                sums["bri_stat"] += float(loss_bri_stat_i.detach().item())
                sums["cbcr"] += float(loss_cbcr_i.detach().item())
                sums["rgb"] += float(loss_rgb_i.detach().item())
                sums["white"] += float(loss_white_i.detach().item())
                sums["flat_tv"] += float(loss_flat_tv_i.detach().item())
                sums["flat_cbcr"] += float(loss_cbcr_flat_i.detach().item())
                sums["sat"] += float(loss_sat_stat_i.detach().item())
                sums["ctr"] += float(loss_ctr_range_i.detach().item())
                sums["cons"] += float(loss_cons_cbcr_i.detach().item())
                sums["y_mean_out"] += float(Y_out.mean().detach().item())
                sums["y_mean_ref"] += float(Y_ref.mean().detach().item())
                sums["y_p95_out"] += float(torch.quantile(Y_out.flatten(), 0.95).detach().item())
                sums["y_p95_ref"] += float(torch.quantile(Y_ref.flatten(), 0.95).detach().item())
                sums["mwhite"] += float(m_white.mean().detach().item())
                sums["mflat"] += float(flat_w.mean().detach().item())
                sums["s_mean"] += float(s_mean_out_i.detach().item())
                mu_out_sum += out.mean(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64)

            # LUT-only regularizers are common to all source images. Adding them once
            # after the averaged data term is identical to averaging per-source total losses.
            loss_s1 = lut.smoothness_loss()
            loss_s2 = lut_second_order_smoothness(lut)
            loss_id = lut.identity_loss()
            loss_lut_luma_boost = lut_luma_boost_limit_loss(
                lut,
                th=lut_luma_boost_th,
                margin=lut_luma_boost_margin,
                topk_ratio=lut_luma_boost_topk_ratio,
                topk_weight=lut_luma_boost_topk_weight,
                max_weight=lut_luma_boost_max_weight,
            )
            regularization_loss = (
                w_lut_luma_boost * loss_lut_luma_boost
                + w_s1 * loss_s1
                + w_s2 * loss_s2
                + w_id * loss_id
            )
            regularization_loss.backward()
            opt.step()

            inv_m = 1.0 / sample_count
            loss_y_value = sums["y"] * inv_m
            loss_bri_stat_value = sums["bri_stat"] * inv_m
            loss_cbcr_value = sums["cbcr"] * inv_m
            loss_rgb_value = sums["rgb"] * inv_m
            loss_white_value = sums["white"] * inv_m
            loss_flat_tv_value = sums["flat_tv"] * inv_m
            loss_cbcr_flat_value = sums["flat_cbcr"] * inv_m
            loss_sat_stat_value = sums["sat"] * inv_m
            loss_ctr_range_value = sums["ctr"] * inv_m
            loss_cons_cbcr_value = sums["cons"] * inv_m
            y_mean_out = sums["y_mean_out"] * inv_m
            y_mean_ref = sums["y_mean_ref"] * inv_m
            y_p95_out = sums["y_p95_out"] * inv_m
            y_p95_ref = sums["y_p95_ref"] * inv_m
            mwhite_ratio = sums["mwhite"] * inv_m
            mflat_ratio = sums["mflat"] * inv_m
            s_mean_out_value = sums["s_mean"] * inv_m
            mu_out = mu_out_sum * inv_m
            mean_data_loss = sums["data_loss"] * inv_m
            loss_value = mean_data_loss + float(regularization_loss.detach().item())

            ecbcr = ema_cbcr.update(loss_cbcr_value)
            eflat = ema_flat.update(loss_flat_tv_value + loss_cbcr_flat_value + 0.5 * loss_white_value)
            ebri = ema_bri.update(loss_y_value + 0.8 * loss_bri_stat_value)
            ectr = ema_ctr.update(loss_ctr_range_value)
            esat = ema_sat.update(loss_sat_stat_value)
            econs = ema_cons.update(loss_cons_cbcr_value)

            score = (
                score_w_cbcr * ecbcr
                + score_w_flat * eflat
                + score_w_bri * ebri
                + score_w_ctr * ectr
                + score_w_sat * esat
                + score_w_cons * econs
                + score_w_lut_luma_boost * float(loss_lut_luma_boost.detach().item())
            )

            if score < best_score - min_delta:
                best_score = score
                save_best(step, best_score)

            if step % log_every == 0:
                step_elapsed = time.perf_counter() - step_start_time
                total_elapsed = time.perf_counter() - total_start_time
                name_text = f"ALL_SRC[{sample_count}]" if mode == "multi_src_single_ref" else sample_names[0]

                logger.log(
                    f"[{step:4d}/{max_steps}] mode={mode_tag} "
                    f"loss={loss_value:.4f} mean_data_loss={mean_data_loss:.4f} score={score:.4f} best={best_score:.4f} "
                    f"CbCr={loss_cbcr_value:.4f} White={loss_white_value:.4f} "
                    f"Y={loss_y_value:.4f} BriStat={loss_bri_stat_value:.4f} "
                    f"FlatTV={loss_flat_tv_value:.4f} FlatCbCr={loss_cbcr_flat_value:.4f} "
                    f"SatStat={loss_sat_stat_value:.4f} CtrRange={loss_ctr_range_value:.4f} "
                    f"LUTLumaBoost={loss_lut_luma_boost.item():.6f}@{w_lut_luma_boost:.4f} "
                    f"ConsCbCr={loss_cons_cbcr_value:.4f}@{w_cons_cbcr:.4f} "
                    f"s1={loss_s1.item():.4f} s2={loss_s2.item():.4f} id={loss_id.item():.4f}@{w_id:.4f} "
                    f"ema(CbCr/Flat/Bri/Ctr/Sat/Cons)=({ecbcr:.4f}/{eflat:.4f}/{ebri:.4f}/{ectr:.4f}/{esat:.4f}/{econs:.4f}) "
                    f"Ymean(out/ref)=({y_mean_out:.4f}/{y_mean_ref:.4f}) "
                    f"Yp95(out/ref)=({y_p95_out:.4f}/{y_p95_ref:.4f}) "
                    f"Mwhite={mwhite_ratio:.3f} Mflat={mflat_ratio:.3f} "
                    f"Smean={s_mean_out_value:.4f} mu_out={mu_out} name={name_text} "
                    f"step_time={format_seconds(step_elapsed)} total_time={format_seconds(total_elapsed)}"
                )

        training_elapsed = time.perf_counter() - total_start_time
        logger.log(f"Training done. Best step={best_step}, best score={best_score:.6f}")
        logger.log("Best LUT files:")
        logger.log(f" - {out_dir / f'BEST_{mode_tag}_{args.tag}_N{N}.cube'}")
        logger.log(f" - {out_dir / f'BEST_{mode_tag}_{args.tag}_N{N}.npy'}")
        logger.log(f"Training time: {format_seconds(training_elapsed)} ({training_elapsed:.3f} seconds)")

        apply_start_time = time.perf_counter()
        applied_output_dir, applied_count = apply_best_lut_to_all_sources(
            lut=lut,
            dataset=dataset,
            mode=mode,
            mode_tag=mode_tag,
            tag=args.tag,
            lut_size=N,
            out_dir=out_dir,
            device=device,
            logger=logger,
        )
        apply_elapsed = time.perf_counter() - apply_start_time
        total_elapsed = time.perf_counter() - total_start_time

        logger.log(f"LUT application time: {format_seconds(apply_elapsed)} ({apply_elapsed:.3f} seconds)")
        logger.log(f"Applied output count: {applied_count}")
        logger.log(f"Applied output directory: {applied_output_dir}")
        logger.log(f"Total time: {format_seconds(total_elapsed)} ({total_elapsed:.3f} seconds)")
        logger.log(f"Log saved to: {log_path}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
