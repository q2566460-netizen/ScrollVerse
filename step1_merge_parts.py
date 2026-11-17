# 文件名: step1_merge_parts.py
import cv2
import glob
import numpy as np

def merge_relay_outpainted_parts(target_h=1080):
    # 按编号顺序读取所有接力扩图导出的分片 part_0.png, part_1.png, ...
    part_files = sorted(glob.glob("part_*.png"), key=lambda x: int(x.split("_")[1].split(".")[0]))
    if len(part_files) < 2:
        raise ValueError("请确保文件夹下至少有 part_0.png 和 part_1.png 两张接力扩图文件！")

    print(f"检测到 {len(part_files)} 张接力分片: {part_files}")

    # 1. 读取所有分片并统一缩放到标准高度 1080px（消除是否点击“超清”带来的尺寸差异）
    parts = []
    for f in part_files:
        img = cv2.imread(f)
        h, w = img.shape[:2]
        scale = target_h / h
        resized = cv2.resize(img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_LANCZOS4)
        parts.append(resized)

    # 2. 以 part_0 为起点，利用模板匹配自动寻找每一张与前一张的精确重叠像素列
    canvas = parts[0]
    for i in range(1, len(parts)):
        next_part = parts[i]
        # 取下一张图左侧 [20px : 180px] 的纵向条带作为匹配模板（避开最左端1~2像素可能存在的边框伪影）
        margin = 20
        strip_w = 160
        template = next_part[:, margin : margin + strip_w]

        # 在当前长卷右侧 60% 的区域内搜索该模板的位置
        search_start_x = int(canvas.shape[1] * 0.4)
        search_region = canvas[:, search_start_x:]

        res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # 计算 next_part 的第 0 列在 canvas 上的精确对齐坐标 align_x
        align_x = search_start_x + max_loc[0] - margin
        overlap_w = canvas.shape[1] - align_x
        print(f"  -> 正在拼接 part_{i}.png: 匹配度={max_val:.4f}, 自动识别重叠区宽度={overlap_w}px")

        # 3. 在重叠区使用余弦渐变羽化，消除 VAE 重编码带来的微小色差
        left_keep = canvas[:, :align_x]
        left_overlap = canvas[:, align_x:].astype(np.float32)
        right_overlap = next_part[:, :overlap_w].astype(np.float32)
        right_new = next_part[:, overlap_w:]

        x_lin = np.linspace(0, np.pi, overlap_w, dtype=np.float32)
        alpha = ((1.0 - np.cos(x_lin)) / 2.0).reshape(1, overlap_w, 1)

        blended_overlap = ((1.0 - alpha) * left_overlap + alpha * right_overlap).astype(np.uint8)
        canvas = np.concatenate([left_keep, blended_overlap, right_new], axis=1)

    cv2.imwrite("master_scroll.png", canvas)
    print(f"拼接完成！已生成无损超高清长卷 master_scroll.png，总尺寸: {canvas.shape[1]} x {canvas.shape[0]}")

if __name__ == "__main__":
    merge_relay_outpainted_parts()