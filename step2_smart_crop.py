# 文件名: step2_smart_crop.py (零重叠无重复切片版)
import cv2
import glob
import json
import numpy as np


def smart_crop_scroll_no_overlap(
    input_path="master_scroll.png",
    target_h=1080,
    num_slices=None,
    search_radius=80,
):
    # 1. 读取原始长卷并统一高度为 1080px (保持宽度为偶数)
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"找不到图片 {input_path}，请确认文件名和路径！")

    orig_h, orig_w = img.shape[:2]
    scale = target_h / orig_h
    total_w = (int(round(orig_w * scale)) // 2) * 2
    master_1080p = cv2.resize(img, (total_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite("master_1080p.png", master_1080p)
    print(f"[1/3] 已生成标准总长卷 master_1080p.png，尺寸: {total_w} x {target_h}")

    # 自动决定切几段：优先匹配 part_*.png 的数量，否则按接近 16:9 的比例自动计算段数
    if num_slices is None:
        part_files = glob.glob("part_*.png")
        if len(part_files) >= 2:
            num_slices = len(part_files)
        else:
            num_slices = max(2, int(round(total_w / 1920.0)))

    # 2. 计算古画笔触密集度，用于在等分点附近微调切线，避免把人物劈成两半
    print(f"[2/3] 正在将长卷无重叠切分为 {num_slices} 段（切片之间 0 重复）...")
    gray = cv2.cvtColor(master_1080p, cv2.COLOR_BGR2GRAY)
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_mag = grad_x + grad_y

    weight_y = np.ones((target_h, 1), dtype=np.float32)
    weight_y[int(target_h * 0.2) : int(target_h * 0.85), 0] = 2.5
    weighted_energy = np.mean(edge_mag * weight_y, axis=0)
    cost_curve = cv2.GaussianBlur(weighted_energy.reshape(1, -1), (51, 1), 0).flatten()

    # 3. 寻找 N-1 条无重叠垂直切线: 0 -> cut_1 -> cut_2 -> ... -> total_w
    cut_points = [0]
    for k in range(1, num_slices):
        nominal_cut = int(round(k * total_w / num_slices))
        s_min = max(cut_points[-1] + 400, nominal_cut - search_radius)
        s_max = min(total_w - 400, nominal_cut + search_radius)

        best_cut = int(s_min + np.argmin(cost_curve[s_min : s_max + 1]))
        # 保证每张切片宽度为偶数，兼容视频编码器
        best_cut = (best_cut // 2) * 2
        cut_points.append(best_cut)
    cut_points.append(total_w)

    # 4. 导出完全不重叠的切片
    slices_info = []
    for idx in range(num_slices):
        x_start = cut_points[idx]
        x_end = cut_points[idx + 1]
        crop_img = master_1080p[:, x_start:x_end]
        fname = f"slice_{idx}.png"
        cv2.imwrite(fname, crop_img)
        slices_info.append(
            {
                "index": idx,
                "file": fname,
                "x_start": int(x_start),
                "x_end": int(x_end),
            }
        )
        print(f"  -> 已导出 {fname}: 范围 [{x_start}, {x_end}] (宽 {x_end - x_start}px，无重叠)")

    with open("slices_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"total_w": total_w, "target_h": target_h, "slices": slices_info},
            f,
            indent=2,
        )
    print(f"[3/3] 零重叠切片完成！共 {num_slices} 张子图，互不重复。")


if __name__ == "__main__":
    smart_crop_scroll_no_overlap()