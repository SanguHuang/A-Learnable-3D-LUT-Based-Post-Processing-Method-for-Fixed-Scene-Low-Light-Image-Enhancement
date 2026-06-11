# v9.8-psnr-boost-no-consistency
# 基于 v9.8-psnr-boost-no-consistency-save-direct 修改。
# 当前训练损失：loss_y、loss_bri_stat、loss_rgb_light、loss_rgb_mse、loss_cbcr、loss_sat_stat、loss_ctr_range、loss_id。
# 已关闭 consistency loss 和 stable gain。
# 已取消训练过程中保存 BEST 直接输出图 PNG，仅保存 BEST LUT 的 .npy 和 .cube 文件。
# 所有训练日志写入 txt 文件，不在命令行中显示；保留训练计时输出。

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
    hi_start: float = 0.82,
    hi_k: float = 8.0,
) -> torch.Tensor:
    # v9.7: 高亮区域权重不再过早压低，避免亮度拟合过于保守。
    w_mid = torch.exp(-((y_ref - mid_center) ** 2) / (2 * (mid_width ** 2))).clamp(0, 1)
    w_hi = torch.sigmoid((hi_start - y_ref) * hi_k)
    return (0.15 + 0.85 * w_mid) * (0.45 + 0.55 * w_hi)


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
    parser.add_argument("--tag", type=str, default="v9_8_psnr_boost_no_consistency", help="Tag used in output file names")
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--lut_size", type=int, default=33)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--log_name", type=str, default=None,
                        help="Optional txt log file name. Default: train_log_<mode>_<tag>_N<lut_size>.txt")
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
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, drop_last=True)
        it = iter(loader)

        logger.log(f"[INFO] mode: {mode}")
        logger.log(f"[INFO] src: {src_path}")
        logger.log(f"[INFO] ref: {ref_path}")
        logger.log(f"[INFO] samples: {len(dataset)}")
        logger.log(f"[INFO] out_dir: {out_dir}")
        logger.log(f"[INFO] log_file: {log_path}")
        logger.log(f"[INFO] device: {device}")


        N = args.lut_size
        lut = LUT3D(N=N, init_identity=True).to(device)
        opt = torch.optim.Adam(lut.parameters(), lr=args.lr)

        # v9.8 PSNR / SSIM boost weights
        # 更强调像素级拟合：提高 Y / RGB 权重，增加 RGB MSE，降低风格统计项权重。
        w_y = 0.90
        w_bri_stat = 0.15
        w_cbcr = 1.60
        w_sat_stat = 0.05
        w_ctr_range = 0.08
        w_rgb_light = 0.22
        w_rgb_mse = 2.00

        # 按要求关闭 consistency loss。
        w_cons_cbcr_start = 0.0
        w_cons_cbcr_end = 0.0
        cons_warmup_end = 1
        cons_anneal_end = 1

        # 恒等约束进一步减弱，仅在前期轻微约束 LUT。
        w_id_start = 0.002
        w_id_end = 0.0
        id_warmup_end = 50
        id_anneal_end = 400

        # 关闭 stable gain，避免单张图拟合时引入额外扰动。
        use_stable_gain = False
        gain_min, gain_max = 1.0, 1.0

        score_w_cbcr = 0.90
        score_w_bri = 0.75
        score_w_bri_stat = 0.25
        score_w_rgb = 0.60
        score_w_rgb_mse = 1.20
        score_w_ctr = 0.12
        score_w_sat = 0.08
        score_w_cons = 0.0
        min_delta = 1e-4

        ema_cbcr = EMA(beta=0.98)
        ema_bri = EMA(beta=0.98)
        ema_bri_stat = EMA(beta=0.98)
        ema_rgb = EMA(beta=0.98)
        ema_rgb_mse = EMA(beta=0.98)
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
            """
            保存当前 BEST LUT。
            只保存 BEST_*.npy 和 BEST_*.cube，不再保存训练过程中的直接输出图 PNG。
            """
            nonlocal best_step, last_best_npy, last_best_cube

            with torch.no_grad():
                lut_grid = lut.lut().detach().cpu()
                lut_rgb = lut_grid.permute(1, 2, 3, 0).numpy()

            npy_path = out_dir / f"BEST_{mode_tag}_{args.tag}_N{N}.npy"
            cube_path = out_dir / f"BEST_{mode_tag}_{args.tag}_N{N}.cube"

            np.save(str(npy_path), lut_rgb)
            save_cube(lut_rgb, str(cube_path), tag=f"fit_lut_v9_8_psnr_boost_no_consistency_{mode_tag}.py")

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

            (src, ref, name), it = _next_batch(it, loader)

            src = src.to(device).clamp(0, 1)
            ref = ref.to(device).clamp(0, 1)

            if use_stable_gain:
                g0 = stable_gain_from_name(name[0], gain_min, gain_max)
                src_in = (src * g0).clamp(0, 1)
            else:
                src_in = src

            out = lut(src_in).clamp(0, 1)

            ycc_out = rgb_to_ycbcr_bt601(out)
            ycc_ref = rgb_to_ycbcr_bt601(ref)

            Y_out, CbCr_out = ycc_out[:, 0:1], ycc_out[:, 1:3]
            Y_ref, CbCr_ref = ycc_ref[:, 0:1], ycc_ref[:, 1:3]

            wy = conservative_y_weight(Y_ref)
            loss_y = (charbonnier(Y_out - Y_ref) * wy).mean()
            loss_bri_stat = brightness_stat_loss_light(Y_out, Y_ref)
            loss_rgb_light = charbonnier(out - ref).mean()
            loss_rgb_mse = F.mse_loss(out, ref)
            loss_cbcr = charbonnier(CbCr_out - CbCr_ref).mean()

            loss_sat_stat, s_mean_out = masked_saturation_stat_loss(out, ref)
            loss_ctr_range = contrast_range_loss(Y_out, Y_ref)

            loss_id = lut.identity_loss()

            w_id = schedule_weight(step, max_steps, w_id_start, w_id_end, id_warmup_end, id_anneal_end)
            w_cons_cbcr = 0.0
            loss_cons_cbcr = torch.zeros((), device=device)

            loss = (
                w_y * loss_y
                + w_bri_stat * loss_bri_stat
                + w_rgb_light * loss_rgb_light
                + w_rgb_mse * loss_rgb_mse
                + w_cbcr * loss_cbcr
                + w_sat_stat * loss_sat_stat
                + w_ctr_range * loss_ctr_range
                + w_id * loss_id
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ecbcr = ema_cbcr.update(float(loss_cbcr.item()))
            ebri = ema_bri.update(float(loss_y.item()))
            ebri_stat = ema_bri_stat.update(float(loss_bri_stat.item()))
            ergb = ema_rgb.update(float(loss_rgb_light.item()))
            ergb_mse = ema_rgb_mse.update(float(loss_rgb_mse.item()))
            ectr = ema_ctr.update(float(loss_ctr_range.item()))
            esat = ema_sat.update(float(loss_sat_stat.item()))
            econs = ema_cons.update(float(loss_cons_cbcr.item()))

            score = (
                score_w_cbcr * ecbcr
                + score_w_bri * ebri
                + score_w_bri_stat * ebri_stat
                + score_w_rgb * ergb
                + score_w_rgb_mse * ergb_mse
                + score_w_ctr * ectr
                + score_w_sat * esat
                + score_w_cons * econs
            )

            if score < best_score - min_delta:
                best_score = score
                save_best(step, best_score)

            if step % log_every == 0:
                y_mean_out = Y_out.mean().detach().cpu().item()
                y_mean_ref = Y_ref.mean().detach().cpu().item()
                y_p95_out = torch.quantile(Y_out.flatten(), 0.95).detach().cpu().item()
                y_p95_ref = torch.quantile(Y_ref.flatten(), 0.95).detach().cpu().item()
                mu_out = out.mean(dim=(0, 2, 3)).detach().cpu().numpy()
                step_elapsed = time.perf_counter() - step_start_time
                total_elapsed = time.perf_counter() - total_start_time

                logger.log(
                    f"[{step:4d}/{max_steps}] mode={mode_tag} "
                    f"loss={loss.item():.4f} score={score:.4f} best={best_score:.4f} "
                    f"CbCr={loss_cbcr.item():.4f} "
                    f"Y={loss_y.item():.4f} BriStat={loss_bri_stat.item():.4f} RGB={loss_rgb_light.item():.4f} RGB_MSE={loss_rgb_mse.item():.6f} "
                    f"SatStat={loss_sat_stat.item():.4f} CtrRange={loss_ctr_range.item():.4f} "
                    f"ConsCbCr={loss_cons_cbcr.item():.4f}@{w_cons_cbcr:.4f} "
                    f"id={loss_id.item():.4f}@{w_id:.4f} "
                    f"ema(CbCr/Y/BriStat/RGB/RGBmse/Ctr/Sat/Cons)=({ecbcr:.4f}/{ebri:.4f}/{ebri_stat:.4f}/{ergb:.4f}/{ergb_mse:.6f}/{ectr:.4f}/{esat:.4f}/{econs:.4f}) "
                    f"Ymean(out/ref)=({y_mean_out:.4f}/{y_mean_ref:.4f}) "
                    f"Yp95(out/ref)=({y_p95_out:.4f}/{y_p95_ref:.4f}) "
                    f"Smean={float(s_mean_out):.4f} mu_out={mu_out} name={name[0]} "
                    f"step_time={format_seconds(step_elapsed)} total_time={format_seconds(total_elapsed)}"
                )

        total_elapsed = time.perf_counter() - total_start_time
        logger.log(f"Done. Best step={best_step}, best score={best_score:.6f}")
        logger.log("Best LUT files:")
        logger.log(f" - {out_dir / f'BEST_{mode_tag}_{args.tag}_N{N}.cube'}")
        logger.log(f" - {out_dir / f'BEST_{mode_tag}_{args.tag}_N{N}.npy'}")
        logger.log(f"Training total time: {format_seconds(total_elapsed)} ({total_elapsed:.3f} seconds)")
        logger.log(f"Log saved to: {log_path}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
