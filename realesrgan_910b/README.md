# H3 → Real-ESRGAN x4plus：昇腾 910B 独立视频超分

不依赖 ComfyUI，不访问 MiniMax 云端，不改变 H3 模型。输入 H3 已生成的视频，
逐帧在指定 NPU 上执行 Real-ESRGAN x4plus，再输出 2K/QHD 或 4K/UHD。
音频采用压缩数据包直接复制，不解码重编码；输出成功前核验音频哈希与时间范围。

**交付状态：已编写 NPU 推理路径并完成本机 CPU/视频封装测试，尚未在真实 910B 上验收。**
不能把 CPU 测试视为 NPU 算子、性能、显存或画质验证。没有改动现有 H3 环境。
权重不包含在程序包中，推理不会自动下载权重。

## 1. 目标服务器准备

使用 Linux + Ascend 910B，先确认驱动、固件、CANN、PyTorch、torch_npu 已配套：

```bash
npu-smi info
# 按实际 CANN 安装位置执行；已有环境可跳过。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c 'import torch,torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.is_available())'
```

复用服务器已有、验证通过的 NPU 软件栈，不为超分盲目升级 H3 的 CANN/PyTorch。
官方配套表：https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md

将本目录复制到服务器，例如 `/opt/realesrgan_910b`。在已能识别 NPU 的 Python 环境中运行：

```bash
cd /opt/realesrgan_910b
bash install.sh
```

如 H3 的可用 Python 不叫 `python3`：

```bash
SR_BASE_PYTHON=/实际路径/bin/python bash install.sh
```

安装脚本只在本目录创建 `.venv`，继承既有 PyTorch/NPU 包，另外安装 NumPy 和 PyAV。
这不是完全隔离的 NPU 栈，基环境升级后应重新验收。不会安装 CUDA、torchvision、
BasicSR、GFPGAN、ComfyUI 或修改 CANN。Python 建议 3.10–3.12，至少 3.10；
PyTorch 需支持 `torch.load(..., weights_only=True)`。

PyAV 提供 FFmpeg 编解码和封装功能，不依赖 `ffmpeg` 命令行。安装后必须能创建
`av.codec.Codec('libx264', 'w')`。aarch64 离线服务器需准备匹配 Python/架构的 PyAV wheel；
没有 wheel 时应按 PyAV 官方文档安装 FFmpeg 开发库后构建，不要复制 Windows/x86 wheel。

## 2. 权重下载（仅需一次）

```bash
.venv/bin/python download_weights.py --output models/RealESRGAN_x4plus.pth
```

来源：https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

也可在联网机器下载后复制到服务器，并在命令中填写绝对路径。只接受官方 x4plus 的
23 个 RRDB、64 特征通道、32 增长通道结构；不接受 anime_6B、x2plus 或任意其他模型。
加载时优先 `params_ema`，严格核验全部参数名称与形状，并使用 `weights_only=True`。

下载脚本打印 SHA-256；生产环境将经可信渠道确认的摘要固定到模型登记表，
推理时用 `--weights-sha256 DIGEST` 校验。自行计算的摘要只能用于后续一致性核验，
不等于第三方来源认证。现有文件一律不覆盖。

## 3. 先做 NPU 小规模检查

```bash
.venv/bin/python check_environment.py \
  --weights models/RealESRGAN_x4plus.pth --device npu:0 --precision fp32
```

检查真实权重加载、NPU 上的卷积/拼接/激活/最近邻上采样、有限值输出，以及视频编码器。
默认 FP32 用于首先建立可验证基线；通过后可用 `--precision fp16` 比较速度和画质。
FP16 不是默认承诺，发现 NaN/Inf 会报错，不能把坏帧写成成功输出。

## 4. 输出 2K 或 4K

本项目将 `2k` 明确定义为 **2560×1440（QHD，严格说不是 DCI 2K）**；
`4k` 为 **3840×2160（UHD）**。不提供插帧，保持原始帧率、帧数和音频。

```bash
# 2K / QHD
.venv/bin/python upscale.py \
  --input /data/h3/input.mp4 \
  --output /data/sr/output_2k.mp4 \
  --weights /data/models/RealESRGAN_x4plus.pth \
  --target 2k --device npu:0

# 4K / UHD
.venv/bin/python upscale.py \
  --input /data/h3/input.mp4 \
  --output /data/sr/output_4k.mp4 \
  --weights /data/models/RealESRGAN_x4plus.pth \
  --target 4k --device npu:0 --tile 256 --overlap 32 --precision fp32
```

默认 `--fit pad` 等比例缩放并补黑边，保留完整内容。H3 的 1344×768 是 7:4：
输出 2560×1440 时画面为 2520×1440，左右各补 20 像素；
输出 3840×2160 时画面为 3780×2160，左右各补 30 像素。
要铺满 16:9，使用 `--fit crop` 居中裁切少量上下画面，不会拉伸脸或身体。

处理顺序是原始画面 **4 倍模型超分 → Lanczos 调整至目标尺寸 → 补边/裁切 → 编码**。
不把 2K 输出再送入模型做 4K；要两个版本，分别从 H3 原片执行。

关键参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | `npu:0` | 当前可见 NPU 的逻辑编号，不自动选择空闲卡 |
| `--precision` | `fp32` | 可选 fp16；CPU 测试只支持 fp32 |
| `--tile` | 256 | 输入图分块边长；OOM 可降到 128 或 64 |
| `--overlap` | 32 | 分块上下文边缘；接缝明显时增加到 64，显存也会上升 |
| `--crf` | 18 | H.264 编码质量，越低体积越大；0 为编码器无损模式但仍经过 YUV420 转换 |
| `--preset` | medium | CPU 编码速度；可改 fast |
| `--fit` | pad | 保留完整画面；crop 为居中裁切 |

全尺寸超分画布放在 CPU 内存，每次仅一个扩边分块送到 NPU。不是整段视频同时入内存，
也不把完整 4x 帧常驻 NPU。输出编码使用 CPU libx264，不需要 NVENC。

## 5. 接在 H3 后面运行

H3 完成并关闭输出文件后，调度器调用：

```bash
bash /opt/realesrgan_910b/after_h3.sh \
  /data/h3/result.mp4 /data/sr/result_4k.mp4 \
  /data/models/RealESRGAN_x4plus.pth 4k npu:0
```

附加参数直接传递，例如末尾添加 `--precision fp16 --fit crop`。
Python 调度器中，在 H3 成功、文件关闭且资源释放后：

```python
import subprocess

# h3_output_path 必须来自你自己的 H3 成功结果，不是硬编码/尚在写入的临时文件。
def upscale_h3_result(h3_output_path, final_output_path):
    subprocess.run([
        "/opt/realesrgan_910b/.venv/bin/python",
        "/opt/realesrgan_910b/upscale.py",
        "--input", str(h3_output_path), "--output", str(final_output_path),
        "--weights", "/data/models/RealESRGAN_x4plus.pth",
        "--target", "4k", "--device", "npu:0",
    ], check=True)
```

**针对你 H3 八卡并行的情况：**

1. 暂停接收新的 H3 任务，等待在途任务完成。
2. 通过已有 H3 管理接口卸载/退出相关 worker，确认所选卡资源已释放。
3. 在其中一张卡启动超分，先单任务运行；完成后进程退出释放模型。
4. 按现有 H3 管理流程恢复服务。

这些管理动作不由本程序擅自执行。本程序没有 H3 的管理接口地址，也不会猜测 PID 或杀进程。
如实测共存有足够余量，可由你的调度器优化，但“卡上剩余 20GB”不是通过验收的保证。
本程序不使用 HCCL，也不会在八卡间拆分一个视频。

同一用户、同一主机、相同设备映射下使用文件锁，防止本程序同时占用同一张卡；
它不锁 H3，不是跨容器/跨用户的全局调度器。保持相同 `ASCEND_RT_VISIBLE_DEVICES` 配置，
否则还需调度器统一物理设备分配。使用可见设备限制时，`npu:0` 是可见集合中的第一张卡。

## 6. 音频与输出安全

- 原始所有音轨直接 remux，不生成新的对白、不做降噪、不改采样率，不缩短到视频长度。
- 无音轨输入保持无音轨。保留音轨 metadata；字幕、附件、章节等不在当前输出范围。
- MP4 不兼容某种原始音频时直接报错，不静默转 AAC；可尝试 `.mkv`，仍必须通过核验。
- 完成后再次解码所有输出帧，核验尺寸、帧数、时间戳；音轨逐条核验 codec、采样率、
  通道数、数据包数量、压缩数据 SHA-256，以及首尾时间（允许 2ms 封装量化）。
- 若容器重封装改变音频数据包或时间，程序拒绝发布结果，不声称保留成功。
- 最终视频通过后才原子发布；已有输入/输出不覆盖。输出目录必须支持硬链接
  （例如 ext4/XFS/NTFS）。FAT/部分网络盘不支持，先输出本地再复制。
- 失败清理本次私有临时目录，不动输入或已有输出；强制断电/SIGKILL 可能留 `.sr-*` 目录。
- 成功后生成 `output.mp4.json` 报告，包含权重摘要、参数、帧数和音频校验结果。
  退出码非零表示未完整完成；不要只凭看到文件就向业务侧报成功。

当前输入范围为 H3 常见的 8-bit SDR、单视频轨、方形像素、从零开始的恒定帧率视频。
可接受 24、25、30、24000/1001 等帧率。可变帧率、旋转、HDR、高位深输入明确拒绝，
避免悄悄改变色彩/时序。大运动或人物细节可能闪烁：Real-ESRGAN 是逐帧超分，没有时序模型。
关闭人脸增强；本程序没有 GFPGAN，避免额外改变身份。

## 7. 测试与验收

```bash
.venv/bin/python -m unittest -v test_upscale
```

本地测试使用合成短视频，验证音频复制、准确目标尺寸、分数帧率、补边/裁切、
分块索引、不覆盖已有文件、异常不发布、VFR 拒绝，以及随机权重 RRDBNet 的 CPU 前向。
真实权重和 NPU 使用 `check_environment.py` 验证；不能用随机权重测试证明超分画质。

服务器验收：先跑一个 1–2 秒真实 H3 片段，比较 FP32/FP16 结果，再测试完整 10 秒。
记录 `npu-smi info` 峰值、耗时及输出报告，检查皮肤、头发、面条等细节是否锐化过度/闪烁。
`tile` 的上下文裁剪只减少接缝，不保证和整帧推理逐像素完全一致。

## 来源

- 模型与权重：https://github.com/xinntao/Real-ESRGAN
- RRDBNet 结构：https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/rrdbnet_arch.py
- PyAV 音轨 remux：https://pyav.basswood.io/docs/stable/cookbook/basics.html#remuxing
- 昇腾版本配套：https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md

`rrdbnet.py` 是 BasicSR 结构的推理专用适配，保留原作者归属和 Apache-2.0 许可证。
未附带原始模型权重。模型许可请同时遵守发布方仓库说明。
