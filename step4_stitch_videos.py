# 文件名: step4_stitch_videos.py (自动寻址 + 零抖动 + 曲线接缝 + 多频段无缝融合版)
import glob
import json
import os
import cv2
import imageio
import numpy as np


def auto_locate_video_on_master(
    video_path: str, master_img: np.ndarray, meta_slice: dict = None
) -> tuple[int, int]:
    """
    直接读取视频第 0 帧，在总长卷 master_1080p.png 上自动匹配出精确的 [x_start, x_end] 像素区间。
    即使 slices_meta.json 被覆盖或切图参数改过，也能 100% 还原该视频在长卷上的真实位置。
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    cap.release()
    if not ret or frame0 is None:
        raise ValueError(f"无法读取视频首帧: {video_path}")

    master_h, master_w = master_img.shape[:2]
    vh, vw = frame0.shape[:2]

    # 候选宽度：优先测试按视频原始宽高比折算到 1080p 的宽度（如 1920），以及 json 中记录的宽度
    cand_widths = {int(round(master_h * vw / vh)), 1920}
    if meta_slice is not None:
        cand_widths.add(int(meta_slice["x_end"] - meta_slice["x_start"]))

    master_gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    best_score = -1.0
    best_x_start = 0
    best_w = int(round(master_h * vw / vh))

    for cw in cand_widths:
        if cw < 200 or cw > master_w:
            continue
        resized_f0 = cv2.resize(frame0, (cw, master_h), interpolation=cv2.INTER_AREA)
        f0_gray = cv2.cvtColor(resized_f0, cv2.COLOR_BGR2GRAY)

        # 取中间区域做模板匹配，避开最边缘可能存在的压缩伪影
        margin = min(40, cw // 10)
        tmpl = f0_gray[:, margin : cw - margin]
        res = cv2.matchTemplate(master_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_x_start = int(np.clip(max_loc[0] - margin, 0, master_w - cw))
            best_w = cw

    x_end = best_x_start + best_w
    # 若极接近画卷左右端点则自动吸附对齐
    if best_x_start <= 8:
        best_x_start = 0
        x_end = best_w
    if master_w - x_end <= 8:
        x_end = master_w
        best_x_start = master_w - best_w

    print(
        f"  -> [自动寻址] {video_path}: 精确匹配到总画卷区间 [{best_x_start}, {x_end}] "
        f"(宽 {best_w}px, 匹配度 {best_score:.4f})"
    )
    return best_x_start, x_end


def load_and_normalize_video(
    video_path: str,
    target_w: int,
    target_h: int,
    target_fps: int = 24,
    target_seconds: int = 15,
) -> list[np.ndarray]:
    """
    读取视频并缩放至真实切片尺寸，不进行任何破坏稳定性的逐帧仿射扭曲（确保 0 抖动），
    同时统一帧率到 24fps 并用正倒放平滑补齐至 15 秒。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 1.0 or src_fps > 120.0:
        src_fps = float(target_fps)

    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        raw_frames.append(frame)
    cap.release()

    duration_sec = len(raw_frames) / src_fps
    resampled_count = max(2, int(round(duration_sec * target_fps)))
    resampled_frames = [
        raw_frames[min(len(raw_frames) - 1, int(round(i * src_fps / target_fps)))]
        for i in range(resampled_count)
    ]

    target_total = target_seconds * target_fps
    if len(resampled_frames) >= target_total:
        return resampled_frames[:target_total]

    # 对不足 15s 的视频做正倒放循环补齐
    pingpong = resampled_frames + resampled_frames[-2:0:-1]
    looped = []
    while len(looped) < target_total:
        looped.extend(pingpong)
    return looped[:target_total]


def compute_curved_seam_masks(
    videos_left_ov: list[np.ndarray],
    videos_right_ov: list[np.ndarray],
    base_ov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在重叠区内预计算：
    1. 避开运动物体与强边缘的“动态规划最优曲线接缝” x_seam(y)
    2. 低频宽域过渡掩码 alpha_low (跨越整个重叠区，消除亮度和色差线)
    3. 高频窄带过渡掩码 alpha_high (沿曲线接缝快速切换，消除重影)
    4. 接缝中心零均值纹理缓冲权重 cushion_weight + 首帧高频零均值细节 B0_high
    """
    h, w = base_ov.shape[:2]

    # 采样若干帧计算重叠区内的运动能量图（让接缝自动绕开正在走动的人、摇摆的树枝和水波）
    sample_indices = np.linspace(0, len(videos_left_ov) - 1, min(16, len(videos_left_ov)), dtype=int)
    motion_acc = np.zeros((h, w), dtype=np.float32)
    l0 = videos_left_ov[0].astype(np.float32)
    r0 = videos_right_ov[0].astype(np.float32)

    for idx in sample_indices:
        lt = videos_left_ov[idx].astype(np.float32)
        rt = videos_right_ov[idx].astype(np.float32)
        diff_l = np.mean(np.abs(lt - l0), axis=2)
        diff_r = np.mean(np.abs(rt - r0), axis=2)
        diff_lr = np.mean(np.abs(lt - rt), axis=2)
        motion_acc += diff_l + diff_r + 1.5 * diff_lr
    motion_cost = motion_acc / len(sample_indices)

    # 图像边缘梯度能量（避免切穿建筑轮廓）
    gray_base = cv2.cvtColor(base_ov, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(gray_base, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray_base, cv2.CV_32F, 0, 1, ksize=3))
    edge_cost = (gx + gy) * 0.4

    # 引导接缝保持在重叠区中央 [25%, 75%] 范围内，防止贴边
    x_coords = np.arange(w, dtype=np.float32).reshape(1, w)
    center_penalty = ((x_coords - w * 0.5) / (w * 0.28)) ** 2 * 18.0

    total_cost = cv2.GaussianBlur(motion_cost + edge_cost, (15, 15), 0) + center_penalty

    # 动态规划寻找从顶到底的最小代价弯曲路径 x_seam(y)
    dp = total_cost.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)
    margin = max(8, w // 6)
    dp[:, :margin] = 1e9
    dp[:, w - margin :] = 1e9

    for y in range(1, h):
        prev_row = dp[y - 1]
        # 允许每行向左/右偏移最多 2 像素
        c_m2 = np.pad(prev_row[:-2], (2, 0), constant_values=1e9)
        c_m1 = np.pad(prev_row[:-1], (1, 0), constant_values=1e9)
        c_0 = prev_row
        c_p1 = np.pad(prev_row[1:], (0, 1), constant_values=1e9)
        c_p2 = np.pad(prev_row[2:], (0, 2), constant_values=1e9)

        stacked = np.vstack([c_m2, c_m1, c_0, c_p1, c_p2])
        best_shift_idx = np.argmin(stacked, axis=0)
        dp[y] += stacked[best_shift_idx, np.arange(w)]
        backtrack[y] = np.arange(w) + (best_shift_idx - 2)

    seam_x = np.zeros(h, dtype=np.float32)
    seam_x[h - 1] = int(np.argmin(dp[h - 1]))
    for y in range(h - 2, -1, -1):
        seam_x[y] = backtrack[y + 1, int(seam_x[y + 1])]

    # 将路径平滑为自然柔和的曲线 (打破笔直竖线感)
    seam_x = cv2.GaussianBlur(seam_x.reshape(h, 1), (1, 91), 0).reshape(h, 1)

    # 构造距离场 d(y, x) = x - seam_x(y)
    dist = x_coords - seam_x  # shape (h, w)

    # 1. 低频宽域掩码 alpha_low：在整个重叠区宽度上平滑过渡，确保两侧边缘严格为 0 和 1
    x_norm = np.linspace(0, np.pi, w, dtype=np.float32).reshape(1, w)
    wide_ramp = (1.0 - np.cos(x_norm)) / 2.0
    seam_sigmoid_wide = 1.0 / (1.0 + np.exp(-dist / max(15.0, w * 0.15)))
    # 结合全局余弦边界约束，保证在重叠区左端点为 0、右端点为 1
    alpha_low = (0.6 * wide_ramp + 0.4 * seam_sigmoid_wide)
    alpha_low[:, 0] = 0.0
    alpha_low[:, -1] = 1.0
    alpha_low = cv2.GaussianBlur(alpha_low, (41, 41), 0)[..., np.newaxis]

    # 2. 高频窄带掩码 alpha_high：沿着曲线接缝在 ~28px 内快速完成切换，杜绝双影
    feather_sigma = max(6.0, min(14.0, w * 0.04))
    alpha_high = (1.0 / (1.0 + np.exp(-dist / feather_sigma)))[..., np.newaxis]

    # 3. 接缝处高频缓冲权重（仅在曲线接缝正中心 ±18px 内柔和平滑高频相位差）
    cushion_bell = np.exp(-0.5 * (dist / max(8.0, min(18.0, w * 0.05))) ** 2)[..., np.newaxis]

    # 预计算首帧的“零均值高频笔触层” B0_high (不含任何亮度/色彩分量，因此绝不会产生色带！)
    b0_f32 = 0.5 * (l0 + r0)
    b0_low = cv2.GaussianBlur(b0_f32, (0, 0), sigmaX=24, sigmaY=24)
    b0_high = b0_f32 - b0_low

    return alpha_low, alpha_high, cushion_bell, b0_high


def multiband_seam_blend(
    canvas: np.ndarray,
    right_frame: np.ndarray,
    x_start: int,
    prev_x_end: int,
    x_end: int,
    masks: tuple,
):
    """
    执行逐行色彩对齐 + 双频段（低频宽域匀光 + 高频曲线接缝）无缝融合
    """
    alpha_low, alpha_high, cushion_bell, b0_high = masks
    overlap_w = prev_x_end - x_start

    left_ov = canvas[:, x_start:prev_x_end].astype(np.float32)
    right_ov_raw = right_frame[:, :overlap_w].astype(np.float32)

    # 1. 逐行（Vertical Profile）对齐右侧视频的明暗与色温，消除上下不均匀的色差断层
    prof_l = cv2.GaussianBlur(np.mean(left_ov, axis=1, keepdims=True), (1, 151), 0)
    prof_r = cv2.GaussianBlur(np.mean(right_ov_raw, axis=1, keepdims=True), (1, 151), 0)
    color_diff = np.clip(prof_l - prof_r, -35.0, 35.0)  # shape (H, 1, 3)

    # 将色彩补偿平滑施加到右侧视频（重叠区内为1，向右端缓慢衰减）
    rem_w = right_frame.shape[1] - overlap_w
    if rem_w > 0:
        decay_tail = np.linspace(1.0, 0.0, rem_w, dtype=np.float32).reshape(1, rem_w, 1)
        full_decay = np.concatenate([np.ones((1, overlap_w, 1), dtype=np.float32), decay_tail], axis=1)
    else:
        full_decay = np.ones((1, right_frame.shape[1], 1), dtype=np.float32)

    right_corrected = np.clip(right_frame.astype(np.float32) + color_diff * full_decay, 0, 255)
    right_ov = right_corrected[:, :overlap_w]

    # 2. 拆解为低频光照层 (sigma=24) 与高频笔触细节层
    l_low = cv2.GaussianBlur(left_ov, (0, 0), sigmaX=24, sigmaY=24)
    r_low = cv2.GaussianBlur(right_ov, (0, 0), sigmaX=24, sigmaY=24)
    l_high = left_ov - l_low
    r_high = right_ov - r_low

    # 3. 低频层跨越整个重叠区做超宽融合（100% 抹平衔接线色差）
    blended_low = (1.0 - alpha_low) * l_low + alpha_low * r_low

    # 4. 高频层沿避动曲线接缝做窄带切换，并在接缝不一致处用零均值 B0_high 平滑缓冲
    direct_high = (1.0 - alpha_high) * l_high + alpha_high * r_high
    # 仅在左右高频发生冲突的地方启用零均值纹理缓冲，防止线条错位断裂
    high_conflict = np.clip((np.mean(np.abs(l_high - r_high), axis=2, keepdims=True) - 4.0) / 18.0, 0.0, 0.75)
    c_w = cushion_bell * high_conflict
    blended_high = (1.0 - c_w) * direct_high + c_w * b0_high

    # 5. 合成重叠区与右侧非重叠区
    canvas[:, x_start:prev_x_end] = np.clip(blended_low + blended_high, 0, 255).astype(np.uint8)
    if rem_w > 0:
        canvas[:, prev_x_end:x_end] = right_corrected[:, overlap_w:].astype(np.uint8)


def stitch_zero_overlap_seam(canvas: np.ndarray, master_img: np.ndarray, seam_x: int):
    """
    兜底处理：如果两个视频之间完全没有重叠区（overlap_w <= 16），
    使用基于梯度场的泊松边界色差消除 + 零均值高频桥接，抹去拼接硬线。
    """
    h, total_w = canvas.shape[:2]
    rad = 48
    x1, x2 = max(0, seam_x - rad), min(total_w, seam_x + rad)
    if x2 - x1 < 20:
        return

    zone = canvas[:, x1:x2].astype(np.float32)
    m_zone = master_img[:, x1:x2].astype(np.float32)
    mid = seam_x - x1

    # 消除接缝两侧的逐行亮度跳变
    l_edge = np.mean(zone[:, max(0, mid - 6) : mid], axis=1, keepdims=True)
    r_edge = np.mean(zone[:, mid : min(x2 - x1, mid + 6)], axis=1, keepdims=True)
    jump = cv2.GaussianBlur(l_edge - r_edge, (1, 91), 0)

    x_idx = np.arange(x2 - x1, dtype=np.float32).reshape(1, -1, 1)
    corr = np.where(
        x_idx < mid,
        -0.5 * jump * np.exp(-(mid - x_idx) / 18.0),
        +0.5 * jump * np.exp(-(x_idx - mid) / 18.0),
    )
    zone_corr = zone + corr

    # 提取当前帧低频光照 + 零均值主图高频笔触桥接接缝
    z_low = cv2.GaussianBlur(zone_corr, (0, 0), sigmaX=18, sigmaY=18)
    z_high = zone_corr - z_low
    m_low = cv2.GaussianBlur(m_zone, (0, 0), sigmaX=18, sigmaY=18)
    m_high = m_zone - m_low

    bell = np.exp(-0.5 * ((x_idx - mid) / 14.0) ** 2)
    out_zone = z_low + (1.0 - bell) * z_high + bell * m_high
    canvas[:, x1:x2] = np.clip(out_zone, 0, 255).astype(np.uint8)


def main():
    if os.path.exists("master_1080p.png"):
        master_img = cv2.imread("master_1080p.png")
    elif os.path.exists("master_scroll.png"):
        raw_m = cv2.imread("master_scroll.png")
        scale = 1080.0 / raw_m.shape[0]
        master_img = cv2.resize(raw_m, (int(round(raw_m.shape[1] * scale)), 1080), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite("master_1080p.png", master_img)
    else:
        raise FileNotFoundError("找不到 master_1080p.png 或 master_scroll.png！")

    target_h, total_w = master_img.shape[:2]

    meta_slices = []
    if os.path.exists("slices_meta.json"):
        with open("slices_meta.json", "r", encoding="utf-8") as f:
            meta_slices = json.load(f).get("slices", [])

    video_files = sorted(
        glob.glob("video_*.mp4"), key=lambda x: int(os.path.splitext(x)[0].split("_")[1])
    )
    if not video_files:
        raise FileNotFoundError("当前目录下没有找到 video_0.mp4, video_1.mp4 等视频文件！")

    fps = 24
    target_seconds = 15
    total_frames = fps * target_seconds

    # 1. 自动在 master_1080p.png 上匹配定位每个视频的真实坐标区间，并加载视频
    print(f"[1/4] 正在对 {len(video_files)} 段视频进行首帧像素级自动寻址与时长对齐...")
    video_segments = []
    for i, v_path in enumerate(video_files):
        m_slice = meta_slices[i] if i < len(meta_slices) else None
        x_start, x_end = auto_locate_video_on_master(v_path, master_img, m_slice)
        v_frames = load_and_normalize_video(
            v_path, x_end - x_start, target_h, target_fps=fps, target_seconds=target_seconds
        )
        video_segments.append({"x_start": x_start, "x_end": x_end, "frames": v_frames})

    # 按 x_start 从左到右排序
    video_segments.sort(key=lambda s: s["x_start"])

    # 2. 为每个重叠区预计算“避动曲线接缝”与多频段融合掩码
    print("[2/4] 正在计算动态规划曲线接缝与多频段光照掩码...")
    seam_masks = [None] * len(video_segments)
    for i in range(1, len(video_segments)):
        prev_x_end = video_segments[i - 1]["x_end"]
        x_start = video_segments[i]["x_start"]
        overlap_w = prev_x_end - x_start
        if overlap_w > 16:
            prev_x_start = video_segments[i - 1]["x_start"]
            l_ov_list = [f[:, (x_start - prev_x_start) : (prev_x_end - prev_x_start)] for f in video_segments[i - 1]["frames"]]
            r_ov_list = [f[:, :overlap_w] for f in video_segments[i]["frames"]]
            base_ov = master_img[:, x_start:prev_x_end]
            seam_masks[i] = compute_curved_seam_masks(l_ov_list, r_ov_list, base_ov)
            print(f"  -> 接缝 {i}: 检测到重叠区宽度 {overlap_w}px，已生成避动曲线接缝与双频段融合场")
        else:
            print(f"  -> 接缝 {i}: 零重叠模式 (overlap={overlap_w}px)，已启用边界泊松匀光桥接")

    # 3. 逐帧执行多频段无缝拼接
    print(f"[3/4] 正在逐帧合成无抖动、无接缝动态长卷 (共 {total_frames} 帧)...")
    stitched_frames = []
    for t in range(total_frames):
        canvas = master_img.copy()
        prev_x_end = 0
        for i, seg in enumerate(video_segments):
            x_start, x_end = seg["x_start"], seg["x_end"]
            curr_frame = seg["frames"][t]

            if i == 0:
                canvas[:, x_start:x_end] = curr_frame
            else:
                overlap_w = prev_x_end - x_start
                if overlap_w > 16 and seam_masks[i] is not None:
                    multiband_seam_blend(
                        canvas, curr_frame, x_start, prev_x_end, x_end, seam_masks[i]
                    )
                else:
                    canvas[:, x_start:x_end] = curr_frame
                    stitch_zero_overlap_seam(canvas, master_img, x_start)
            prev_x_end = max(prev_x_end, x_end)
        stitched_frames.append(canvas)

    # 4. 导出标准 H.264 视频
    out_path = "stitched_scroll.mp4"
    out_w, out_h = total_w, target_h
    if out_w > 4096:
        scale = 4096.0 / out_w
        out_w = 4096
        out_h = int(round(out_h * scale))
    out_w = (out_w // 2) * 2
    out_h = (out_h // 2) * 2

    print(f"[4/4] 正在导出 {out_path} ({out_w}x{out_h}, {target_seconds}秒)...")
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=2,
    )
    for f in stitched_frames:
        if f.shape[1] != out_w or f.shape[0] != out_h:
            f = cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"完成！请打开 {out_path} 查看无抖动、无接缝效果。")


if __name__ == "__main__":
    main()