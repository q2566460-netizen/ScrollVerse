# Dynamic Ancient Chinese Scroll Generator (动态中国古风长卷生成管线)

本项目旨在构建一个生成动态中国古代风格长卷（如《清明上河图》《千里江山图》风格）的自动化渲染管线与智能体。通过将 **AI 局部微观动态生成（I2V）** 与 **Blender 3D 全局物理卷轴匀速展开** 解耦，实现高分辨率、零拖影、严格匀速展开的古画长卷视频生成。

## 项目架构与工作流

1. **Step 1 - 留尾接力扩图拼接 (`step1_merge_parts.py`)**：突破单次扩图分辨率与画幅限制，基于模板匹配与余弦羽化合成超宽无损静态底图 `master_scroll.png`。
2. **Step 2 - 自适应避让切片 (`step2_smart_crop.py`)**：基于古画笔触梯度能量分析，自动避开人物、车马与船只主体切分分镜子图。
3. **Step 3 - 局部固定机位图生视频 (Seedance I2V)**：对各分镜独立生成高精细度局部微动视频（人物行走、水波流动、树枝摇摆）。
4. **Step 4 - 曲线接缝与多频段视频合成 (`step4_stitch_videos.py`)**：首帧金字塔自动寻址对齐，采用动态规划寻找避动曲线接缝，结合低频宽域匀光与高频窄带过渡消除拼接线。
5. **Step 5 - Blender 3D 卷轴匀速展开渲染 (`step5_blender_scroll.py`)**：全自动构建 3D 卷轴与桌面场景，强制线性插值 (`LINEAR`) 驱动卷轴旋转剥离与摄像机匀速横移。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 依次运行管线

```bash
# 1. 将接力扩图分片 (part_0.png, part_1.png...) 合成为超宽底图
python step1_merge_parts.py

# 2. 智能避让切分生成分镜子图 (slice_0.png, slice_1.png...)
python step2_smart_crop.py

# 3. 将图生视频结果 (video_0.mp4, video_1.mp4...) 无缝合成为超宽动态贴图
python step4_stitch_videos.py

# 4. 调用 Blender 渲染最终卷轴展开视频
blender -b -P step5_blender_scroll.py
```

## 开发进度 (Roadmap)

- [x] 超宽长卷无损接力扩图与自适应切片算法
- [x] 多时长/多帧率视频曲线接缝与双频段无缝拼接
- [x] Blender Python (`bpy`) 自动化 3D 卷轴建模与线性匀速渲染
- [ ] **Step 7 (进行中)**：接入 LLM 导演规划与图像/视频生成 API，封装为端到端自动化智能体 (Agent)