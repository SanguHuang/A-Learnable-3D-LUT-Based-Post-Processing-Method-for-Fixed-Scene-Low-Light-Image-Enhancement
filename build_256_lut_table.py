#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 3DLUT（.npy 或 .cube）离线展开为 256^3 RGB 直接查找表。

用途：
    输入一个 33^3 等尺寸的 3DLUT，通过三线性插值预计算所有 8-bit RGB 输入
    对应的输出 RGB，生成 table_256_rgb_uint8.bin。

生成的表索引方式：
    idx = (R << 16) | (G << 8) | B
    out_R = table[idx * 3 + 0]
    out_G = table[idx * 3 + 1]
    out_B = table[idx * 3 + 2]

示例：
    python build_256_lut_table.py --input best_lut.npy --output table_256_rgb_uint8.bin
    python build_256_lut_table.py --input style.cube --output table_256_rgb_uint8.bin

可选：
    --batch 1000000          每批处理的 RGB 数量，内存不够可调小
    --save-npy              同时保存 .npy 格式，方便 Python 检查
    --output-npy table.npy  指定 .npy 输出路径
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np


def load_lut_npy(path: str) -> np.ndarray:
    """读取 .npy LUT，支持 [K,K,K,3] 或 [3,K,K,K]。"""
    lut = np.load(path)

    if lut.ndim != 4:
        raise ValueError(f"npy LUT 维度应为 4，但得到 shape={lut.shape}")

    # 常见格式 1: [K, K, K, 3]
    if lut.shape[-1] == 3 and lut.shape[0] == lut.shape[1] == lut.shape[2]:
        lut = lut.astype(np.float32)
    # 常见格式 2: [3, K, K, K]
    elif lut.shape[0] == 3 and lut.shape[1] == lut.shape[2] == lut.shape[3]:
        lut = np.transpose(lut, (1, 2, 3, 0)).astype(np.float32)
    else:
        raise ValueError(
            "无法识别 npy LUT 形状。请使用 [K,K,K,3] 或 [3,K,K,K]，"
            f"当前 shape={lut.shape}"
        )

    return normalize_lut_range(lut, source=os.path.basename(path))


def load_lut_cube(path: str) -> np.ndarray:
    """
    读取 .cube LUT。

    说明：
    - 支持常见 3D .cube 文件。
    - 输出统一为 [K, K, K, 3]，值域 [0,1]。
    - 大多数 .cube 数据顺序为 B 变化最快，其次 G，最后 R。
    """
    size = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            key = parts[0].upper()

            if key == "TITLE":
                continue
            elif key == "LUT_3D_SIZE":
                if len(parts) < 2:
                    raise ValueError("LUT_3D_SIZE 行格式错误")
                size = int(parts[1])
            elif key == "DOMAIN_MIN":
                if len(parts) >= 4:
                    domain_min = np.array(list(map(float, parts[1:4])), dtype=np.float32)
            elif key == "DOMAIN_MAX":
                if len(parts) >= 4:
                    domain_max = np.array(list(map(float, parts[1:4])), dtype=np.float32)
            elif key.startswith("LUT_1D"):
                raise ValueError("当前脚本只支持 3D .cube，不支持 1D LUT")
            else:
                if len(parts) >= 3:
                    try:
                        rgb = [float(parts[0]), float(parts[1]), float(parts[2])]
                        values.append(rgb)
                    except ValueError:
                        # 忽略未知头部字段
                        pass

    if size is None:
        raise ValueError("未在 .cube 文件中找到 LUT_3D_SIZE")

    expected = size ** 3
    if len(values) != expected:
        raise ValueError(
            f".cube 数据数量不匹配：期望 {expected} 行 RGB，实际 {len(values)} 行"
        )

    arr = np.asarray(values, dtype=np.float32)

    # .cube 常见排列：for r in R: for g in G: for b in B: 写一行，B 变化最快
    lut = arr.reshape(size, size, size, 3)

    # 若 DOMAIN 不是 [0,1]，一般 LUT 输出值本身仍常在 [0,1]。
    # 这里仅对输出值做范围规范化，不强行根据 DOMAIN_MIN/MAX 变换输出。
    # DOMAIN_MIN/MAX 更多描述输入域。
    _ = domain_min, domain_max

    return normalize_lut_range(lut, source=os.path.basename(path))


def normalize_lut_range(lut: np.ndarray, source: str = "LUT") -> np.ndarray:
    """将 LUT 值域规范到 [0,1]。"""
    lut = lut.astype(np.float32)

    min_v = float(np.nanmin(lut))
    max_v = float(np.nanmax(lut))

    if not np.isfinite(min_v) or not np.isfinite(max_v):
        raise ValueError(f"{source} 中包含 NaN 或 Inf")

    # 如果像是 0~255 的 LUT，自动除以 255。
    if max_v > 2.0:
        print(f"[INFO] 检测到 {source} 最大值 {max_v:.6f}，按 0~255 LUT 处理并除以 255。")
        lut = lut / 255.0

    # 轻微越界直接截断。
    if min_v < 0.0 or max_v > 1.0:
        print(f"[WARN] {source} 值域为 [{min_v:.6f}, {max_v:.6f}]，将 clip 到 [0,1]。")
        lut = np.clip(lut, 0.0, 1.0)

    if lut.shape[0] < 2:
        raise ValueError("LUT 尺寸 K 至少为 2")

    return lut.astype(np.float32, copy=False)


def load_lut(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return load_lut_npy(path)
    if ext == ".cube":
        return load_lut_cube(path)
    raise ValueError("只支持 .npy 或 .cube 文件")


def trilinear_lut_batch(lut: np.ndarray, rgb_u8: np.ndarray) -> np.ndarray:
    """
    对一批 8-bit RGB 输入执行三线性 LUT 插值。

    参数：
        lut: [K,K,K,3], float32, [0,1]
        rgb_u8: [N,3], uint8，通道顺序为 RGB

    返回：
        out_u8: [N,3], uint8，通道顺序为 RGB
    """
    K = lut.shape[0]

    rgb = rgb_u8.astype(np.float32) / 255.0
    pos = rgb * (K - 1)

    idx0 = np.floor(pos).astype(np.int32)
    idx0 = np.clip(idx0, 0, K - 2)
    idx1 = idx0 + 1

    d = pos - idx0.astype(np.float32)
    dr = d[:, 0:1]
    dg = d[:, 1:2]
    db = d[:, 2:3]

    r0, g0, b0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
    r1, g1, b1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]

    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    out = (
        c000 * (1.0 - dr) * (1.0 - dg) * (1.0 - db) +
        c001 * (1.0 - dr) * (1.0 - dg) * db +
        c010 * (1.0 - dr) * dg * (1.0 - db) +
        c011 * (1.0 - dr) * dg * db +
        c100 * dr * (1.0 - dg) * (1.0 - db) +
        c101 * dr * (1.0 - dg) * db +
        c110 * dr * dg * (1.0 - db) +
        c111 * dr * dg * db
    )

    out_u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out_u8


def build_table_256(lut: np.ndarray, output_bin: str, batch: int = 1_000_000) -> np.ndarray:
    """
    生成并写出 256^3 直接查找表。

    返回内存中的 table，shape=[16777216,3]，dtype=uint8。
    """
    total = 256 ** 3
    table = np.empty((total, 3), dtype=np.uint8)

    print(f"[INFO] LUT shape: {lut.shape}, dtype={lut.dtype}, range=[{lut.min():.6f}, {lut.max():.6f}]")
    print(f"[INFO] 开始生成 256^3 直接查找表，共 {total:,} 个 RGB 输入。")
    print(f"[INFO] 输出表大小约 {table.nbytes / 1024 / 1024:.2f} MB")

    for start in range(0, total, batch):
        end = min(start + batch, total)
        idx = np.arange(start, end, dtype=np.uint32)

        # 索引约定：idx = (R << 16) | (G << 8) | B
        R = ((idx >> 16) & 255).astype(np.uint8)
        G = ((idx >> 8) & 255).astype(np.uint8)
        B = (idx & 255).astype(np.uint8)
        rgb_u8 = np.stack([R, G, B], axis=1)

        table[start:end] = trilinear_lut_batch(lut, rgb_u8)

        done = end / total * 100.0
        print(f"\r[INFO] 进度: {done:6.2f}% ({end:,}/{total:,})", end="", flush=True)

    print("\n[INFO] 生成完成，正在写入 bin 文件...")
    table.tofile(output_bin)
    print(f"[OK] 已保存: {output_bin}")

    return table


def write_cpp_header_example(path: str) -> None:
    code = r'''// 256^3 RGB uint8 直接查找表示例用法
// 表文件由 build_256_lut_table.py 生成，大小应为 256*256*256*3 字节。

#include <cstdint>
#include <vector>
#include <fstream>
#include <stdexcept>

std::vector<uint8_t> load_table_256_rgb(const char* path) {
    const size_t table_size = 256u * 256u * 256u * 3u;
    std::vector<uint8_t> table(table_size);

    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("failed to open table file");
    }
    fin.read(reinterpret_cast<char*>(table.data()), table.size());
    if (static_cast<size_t>(fin.gcount()) != table_size) {
        throw std::runtime_error("table file size mismatch");
    }
    return table;
}

inline void apply_table_one_pixel_rgb(
    const uint8_t* table,
    uint8_t R, uint8_t G, uint8_t B,
    uint8_t& outR, uint8_t& outG, uint8_t& outB
) {
    uint32_t idx = (static_cast<uint32_t>(R) << 16) |
                   (static_cast<uint32_t>(G) << 8)  |
                   static_cast<uint32_t>(B);
    const uint32_t base = idx * 3u;
    outR = table[base + 0];
    outG = table[base + 1];
    outB = table[base + 2];
}

void apply_table_image_rgb(
    const uint8_t* input_rgb,
    uint8_t* output_rgb,
    int width,
    int height,
    const uint8_t* table
) {
    const int pixels = width * height;
    for (int i = 0; i < pixels; ++i) {
        uint8_t R = input_rgb[i * 3 + 0];
        uint8_t G = input_rgb[i * 3 + 1];
        uint8_t B = input_rgb[i * 3 + 2];

        uint32_t idx = (static_cast<uint32_t>(R) << 16) |
                       (static_cast<uint32_t>(G) << 8)  |
                       static_cast<uint32_t>(B);
        uint32_t base = idx * 3u;

        output_rgb[i * 3 + 0] = table[base + 0];
        output_rgb[i * 3 + 1] = table[base + 1];
        output_rgb[i * 3 + 2] = table[base + 2];
    }
}
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 256^3 uint8 RGB direct lookup table from 3DLUT npy/cube.")
    parser.add_argument("--input", "-i", required=True, help="输入 3DLUT 文件，支持 .npy 或 .cube")
    parser.add_argument("--output", "-o", default="table_256_rgb_uint8.bin", help="输出 bin 文件路径")
    parser.add_argument("--batch", type=int, default=1_000_000, help="每批处理 RGB 数量，内存不足可调小")
    parser.add_argument("--save-npy", action="store_true", help="是否额外保存 .npy 文件")
    parser.add_argument("--output-npy", default="table_256_rgb_uint8.npy", help="额外保存的 .npy 路径")
    parser.add_argument("--write-cpp-example", default="apply_table_256_example.cpp", help="写出 C++ 查表示例代码路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.batch <= 0:
        raise ValueError("--batch 必须大于 0")

    lut = load_lut(args.input)
    table = build_table_256(lut, args.output, batch=args.batch)

    if args.save_npy:
        print(f"[INFO] 正在保存 npy: {args.output_npy}")
        np.save(args.output_npy, table)
        print(f"[OK] 已保存: {args.output_npy}")

    if args.write_cpp_example:
        write_cpp_header_example(args.write_cpp_example)
        print(f"[OK] 已写出 C++ 示例: {args.write_cpp_example}")

    print("[DONE] 全部完成。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
