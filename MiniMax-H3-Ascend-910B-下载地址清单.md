# MiniMax-H3 官方版 + Ascend 910B：纯下载地址 Checklist

更新时间：2026-09-01

## 一、MiniMax-H3（必需）

- [ ] 官方代码仓库  
  <https://github.com/MiniMax-AI/MiniMax-H3>

- [ ] 官方模型权重总目录  
  <https://huggingface.co/MiniMaxAI/MiniMax-H3>

- [ ] FL2VA 权重目录——文生视频、首帧/尾帧生视频必需  
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA>

- [ ] Ref2VA 权重目录——图片、视频、音频参考生成时才需要  
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/Ref2VA>

- [ ] 根目录 `model_index.json`  
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/model_index.json>

- [ ] MiniMax-H3 Community License  
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE>

> `FL2VA` 和 `Ref2VA` 目录内已经包含各自所需的 processor、tokenizer、text encoder、transformer、Visual VAE 和 Audio VAE；采用官方原始 checkpoint 时不需要再单独寻找 Qwen3-VL、视频 VAE 或音频 VAE。

## 二、Ascend 910B 基础软件（必需）

- [ ] Ascend 910B 驱动与固件——HDK 25.5.2 入口  
  <https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=9.0.0&driver=Ascend+HDK+25.5.2>

- [ ] CANN 社区版下载中心  
  <https://www.hiascend.com/developer/download/community/result?module=cann>

- [ ] CANN 9.0.0 安装资料  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0008.html>

- [ ] Ascend PyTorch / torch-npu 下载与版本匹配  
  <https://ascend.github.io/docs/sources/pytorch/install.html>

- [ ] torch-npu 官方发布页  
  <https://gitcode.com/Ascend/pytorch/releases>

推荐的 910B Docker 版本组：HDK 25.5.2、CANN 9.0.0、Python 3.11、PyTorch 2.10.0、torch-npu 2.10.0。不要与 CANN 9.1.0 版本组交叉混装。

## 三、SGLang Ascend 推理环境（必需）

- [ ] SGLang 官方仓库  
  <https://github.com/sgl-project/sglang>

- [ ] MiniMax-H3 的 SGLang 官方部署页  
  <https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3>

- [ ] SGLang Ascend NPU 安装及版本对应表  
  <https://docs.sglang.io/docs/hardware-platforms/ascend-npus/getting-started/installation>

- [ ] SGLang NPU Kernel 2026.05.01.post3  
  <https://github.com/sgl-project/sgl-kernel-npu/releases/tag/2026.05.01.post3>

- [ ] Triton Ascend  
  <https://github.com/triton-lang/triton-ascend>

- [ ] SGLang 910B 日构建 Docker 镜像地址  
  `quay.io/ascend/sglang:main-cann9.0.0-910b`

- [ ] SGLang 910B 稳定版 Docker 镜像地址  
  `quay.io/ascend/sglang:cann9.0.0-910b-v0.5.16`

- [ ] 910B CANN 9.0.0 基础 Docker 镜像地址——仅源码构建环境时需要  
  `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.11`

> MiniMax-H3 的 Ascend Diffusion 支持较新，优先选择 `main-cann9.0.0-910b`；验证后记录具体镜像 digest。稳定版是否包含所需的 H3 Diffusion 功能，应以容器内版本为准。

## 四、通用工具（必需）

- [ ] Docker Engine  
  <https://docs.docker.com/engine/install/>

- [ ] Hugging Face CLI  
  <https://huggingface.co/docs/huggingface_hub/guides/cli>

- [ ] FFmpeg / ffprobe  
  <https://ffmpeg.org/download.html>

- [ ] Git  
  <https://git-scm.com/downloads>

- [ ] Git LFS  
  <https://git-lfs.com/>

## 五、ComfyUI（可选）

- [ ] ComfyUI 官方仓库  
  <https://github.com/Comfy-Org/ComfyUI>

- [ ] ComfyUI Ascend NPU 安装说明  
  <https://github.com/Comfy-Org/ComfyUI#ascend-npus>

- [ ] ComfyUI MiniMax-H3 官方教程  
  <https://docs.comfy.org/tutorials/video/minimax/minimax-h3>

- [ ] MiniMax-H3 T2V 工作流模板  
  <https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json>

- [ ] MiniMax-H3 R2V 工作流模板  
  <https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json>

> ComfyUI 官方 H3 模板通常使用 Comfy-Org 转换权重，不是上述 MiniMax 官方原始 checkpoint。若采用“SGLang 在 910B 上加载官方权重，ComfyUI 只负责界面调用”的架构，则不需要再下载 Comfy-Org 权重。

## 最小下载集合

只做文生视频时，最小集合是：

1. MiniMax-AI/MiniMax-H3 GitHub 仓库。
2. Hugging Face 根 `model_index.json` 和完整 `FL2VA` 目录。
3. Ascend 910B HDK 25.5.2 驱动/固件。
4. `quay.io/ascend/sglang:main-cann9.0.0-910b` 镜像。
5. Docker、Hugging Face CLI、FFmpeg。

