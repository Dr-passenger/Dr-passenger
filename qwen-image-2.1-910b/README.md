# Qwen-Image-2.1 / Ascend 910B：与八卡 H3 错峰运行

这是一个 Linux 命令行部署包。**模型路径按要求留空；不包含、不自动下载权重。**

目标：保留 H3 八卡模型常驻与现有审核模型，用第8张卡暂时运行 Qwen BF16 文生图。
H3 不改成七卡，不杀 H3 进程，不关闭审核。当前版本只开放单请求、单张文生图；不开放图像编辑、批量、HTTP 公网服务或 ComfyUI 节点。

## 验证边界（先读）

- 已进行 Python 语法检查与不依赖 NPU 的调度/失败恢复单元测试。
- **尚未在你的 910B 服务器上完成安装、算子适配、峰值测量或真实出图。**
- 使用官方 `QwenImage21Pipeline` + `torch_npu` 原生 NPU 设备 + Accelerate 卸载接口；不是把 CUDA 字符串替换一下就声称适配完成。
- 你现有 H3 的调度接口未知，因此 `h3_adapter.py` 特意默认报错，不能伪造“已暂停”。接入真实 H3 调度器是上线前必需的一步。
- 模型路径、物理卡映射、H3 控制适配器任何一项缺失，程序拒绝生成。
- 这里的 GB 是权重发布页的十进制大小；配置和测量统一使用 GiB（1024³字节）。

## 1. 运行顺序

1. 本程序获取本机调度锁，写入持久化租约日志。
2. H3 调度器暂停派发新推理，但保持八卡进程、权重、审核服务运行。
3. 等待 H3 **所有来源、所有模型分支**的在途推理结束；排队任务保持等待。
4. 探测选定 NPU 的可用显存和主机可用内存，记录显存基线。
5. 启动一个独立 Qwen 子进程，只暴露指定卡，按编码器→生成模型→VAE 分阶段卸载。
6. 保存图片和统计，结束 Qwen 子进程。进程退出是资源释放边界，不让 Qwen 长期占住卡8。
7. 再次探测显存，回到基线容差范围后，用原租约恢复 H3 派发。

因此每次请求会重新从本地文件加载 Qwen。该版本优先可控释放与故障隔离，不承诺 CPU 权重热驻留；操作系统文件缓存可能加快重复加载。

**不能仅轮询 ComfyUI 的队列是否为空。** 必须先原子关闭所有 H3 任务的派发入口，再等待在途任务归零，才能消除检查后又启动新任务的竞态。

## 2. 文件

| 文件 | 用途 |
|---|---|
| `config.json` | 模型路径、卡号、资源预算、H3控制命令 |
| `coordinator.py` | 暂停/排空/生成/释放/恢复、持久化租约 |
| `worker.py` | 一次性 BF16 NPU 文生图子进程 |
| `h3_adapter.py` | 必须接入真实 H3 调度器的适配入口；默认拒绝运行 |
| `install.py` | 新建隔离虚拟环境，锁住现有 torch/torch-npu 版本 |
| `request.example.json` | 文生图请求示例 |
| `tests/test_scheduler.py` | 不使用 NPU 的单元测试 |

## 3. 准备独立运行环境

把本目录复制到服务器，例如 `/opt/qwen-image-2.1-910b`。

推荐在**独立的 Ascend 容器/环境**中部署，使用与你的910B驱动匹配的 CANN、PyTorch、torch-npu 和 Python 3.10+。也可以用已有可工作的 Ascend Python 创建独立虚拟环境，但不要往 H3 的环境直接 `pip install -U`。

先由管理员按实际安装位置加载 CANN 环境变量，并确认 `npu-smi info` 正常。脚本不安装驱动、不更换 CANN、不升级宿主 H3。

选择一个已审阅、包含 `QwenImage21Pipeline` 的 Diffusers 完整提交 SHA，然后执行：

```bash
cd /opt/qwen-image-2.1-910b
python3 install.py --diffusers-ref <已审阅的40位Diffusers提交SHA>
```

尖括号是占位说明，不要原样执行。可在官方源码仓库查找对应提交，或在已审阅的源码检出目录运行 `git rev-parse HEAD`。

安装器要求当前 Python 已能找到兼容的 `torch` 与 `torch-npu`：

- 新建 `.venv`，继承基础 Ascend 包，约束 torch 配对版本不变化；不会覆盖基础环境。
- 在新虚拟环境安装 `requirements.in` 和指定 Diffusers 提交。
- 执行 `pip check`、导入检查，并保存 `requirements.resolved.txt` 和 `torch.constraints.txt`。
- 如果现有 torch 配对与新依赖不兼容，停止并选择独立的兼容 Ascend 环境，不要强行更新 H3。
- `.venv` 已存在时拒绝覆盖；升级请使用新的部署目录。
- 安装依赖需要联网，或自行提供内部镜像/离线包。实际推理强制使用离线模型文件。

上游接口会变化，因此不凭空指定一组“已在910B验证”的版本。安装器锁定 Diffusers 提交并记录解析后的版本；完整硬件兼容仍以服务器测试为准。

## 4. 填配置，模型路径当前留空

`config.json` 中初始值为：

```json
"model_path": "",
"physical_device_id": "",
"device_mapping_confirmed": false,
"h3_control_argv": []
```

准备好后填写：

- `model_path`：**Qwen/Qwen-Image-2.1 的完整本地 Diffusers 目录绝对路径**。必须有 `model_index.json`、`transformer/`、`text_encoder/`、`vae/`、`scheduler/`、`processor/` 等配套文件。不接受只有 Comfy-Org 三个单文件权重的目录。
- `physical_device_id`：核对 `npu-smi info` 和容器设备映射后填写。人称“第8张卡”常对应ID `7`，**但不能直接假定**。程序通过 `ASCEND_RT_VISIBLE_DEVICES` 限制可见卡，进程内部用 `npu:0`；若可见卡不恰好一张就报错。容器也应只映射目标卡及运行所需管理设备。
- `device_mapping_confirmed`：人工确认对应那张剩余约29GB的卡后改为 `true`。
- `h3_control_argv`：例如 `["/opt/h3-control/bin/python", "/opt/h3-control/h3_adapter.py"]`。用**绝对路径和参数数组**，不使用shell拼接。适配器可用独立的H3控制环境，不必和Qwen共用Python。

不要把示例卡号当成实际检查结果；本程序无法远程识别你的卡8。

初始容量限制：

| 配置 | 默认 | 含义 |
|---|---:|---|
| `min_free_npu_gib` | 24 | 开始加载前至少需要的全设备空闲量 |
| `reserve_npu_gib` | 5 | 计划保留的显存余量 |
| `max_qwen_allocator_gib` | 24 | Qwen 本进程 PyTorch 缓存分配器上限 |
| `min_available_host_gib` | 64 | Linux MemAvailable 最低值，不是机器总内存 |
| `max_pixels` | 1048576 | 初始最大面积1024×1024 |
| `offload_mode` | `model` | 组件级CPU卸载；`sequential`更省显存但更慢 |
| `release_tolerance_gib` | 1 | 子进程退出后允许相对基线的波动 |

实际分配器预算取 `min(24GiB, 当时空闲量−5GiB)`。这是资源限制，不是运行峰值承诺。
`set_per_process_memory_fraction` **不能限制所有CANN工作区/驱动内存**；采样回调也不是连续硬实时保护。其他进程（包括审核）仍可能分配显存。必须实测并留足余量。

## 5. 接入 H3 适配器（必须）

本包不能猜测你已有服务的URL、认证、任务统计方式；`h3_adapter.py`是唯一需要按现场H3系统接线的位置，不是可直接运行的演示后端。

调度器会执行：

```text
<h3_control_argv...> pause  <lease_id>
<h3_control_argv...> status <lease_id>
<h3_control_argv...> resume <lease_id>
```

适配器成功退出码为0，标准输出**仅输出一个JSON对象**，日志写标准错误。失败退出非0。

### pause

原子取得带所有者的持久化暂停租约，阻止后续H3派发，不中断当前生成，不卸载权重。成功返回：

```json
{"lease_id":"原样返回传入ID","paused":true}
```

### status

确认当前暂停仍属于这个租约，返回所有H3推理实例的在途任务总数。所有异步NPU工作完成后才允许归零：

```json
{"lease_id":"原样返回传入ID","paused":true,"active_jobs":0}
```

### resume

只释放此租约，不释放其他运维暂停、不更改原有审核要求。成功返回：

```json
{"lease_id":"原样返回传入ID","resumed":true}
```

### 调度器必须保证

- 所有API、ComfyUI、定时任务、直接调用入口均受同一个派发闸门约束；不能有绕过入口。
- 多租约互斥/所有权检查；同一租约重试必须幂等，不得偷取另一作业的暂停。
- 暂停持久化，不能因Qwen父进程异常退出、H3调度器重启或简单TTL过期而自动放行。
- active_jobs计数与派发在同一原子控制机制内；不能仅根据NPU利用率为0推断空闲。
- 不得把 `pause` 实现为 `kill`、停止H3容器、关闭审核模型或清空队列。
- 本地 `coordinator.lock` 只防止本包的多实例重入，**不会自动锁住现有H3服务**。跨容器实例应共用同一持久化state目录，且必须由H3调度器全局协调。

如果H3暂时没有此类能力，先实现这层闸门再上线；不要把适配器改成固定输出成功JSON。

审核服务维持原样；本包输出的是**待审核中间文件**，不会自动提交给H3、不会对外发布。需要在原有业务链中审核提示词/图片，通过后再交给H3。单纯保留审核模型进程并不等于新Qwen请求已经过审核。

## 6. 检查与生成

```bash
cd /opt/qwen-image-2.1-910b
.venv/bin/python coordinator.py check
.venv/bin/python coordinator.py generate --request request.example.json
```

`check`只检查配置与模型目录结构，不触碰H3或NPU，不保证权重完整性或算子兼容。
生产环境应在下载阶段另外核验模型文件清单/哈希。实际首次生成会验证导入、单卡可见性、BF16小矩阵计算和完整推理。

输出目录 `outputs/<lease_id>/`：

- `image.png`：生成的待审核图片。
- `request.json`：输入参数。
- `worker.log`：初始化、采样进度和异常。
- `metrics.json`：加载时间、推理时间、本进程分配器峰值、依赖版本。
- `result.json`：完成/失败和H3恢复结果；只有完整释放/恢复流程成功后才写入。
- `config.resolved.json`：当次配置，便于追踪；不要在控制命令参数中嵌入密钥，日志目录不应公开。

第一轮建议在维护窗口完成：单张1024出图→确认H3恢复→最重H3视频回归→重复切换。
需要2K时，只有经过峰值测试才提高 `max_pixels` 至4194304；不要同时开放批量。若OOM，先查看日志，保留5GiB以上余量，再考虑 `sequential`；它也需要在910B上验证且通常更慢。

程序默认会在一小时后终止**自己启动的Qwen子进程组**，不会终止H3。可按实测调整 `worker_timeout_seconds`。

## 7. 失败与恢复

正常推理异常/OOM：等待Qwen退出→验证显存恢复→恢复H3→向调用者报告失败。

以下情况保留 `runtime/active.json` 并拒绝下一次生成：暂停响应不明、H3排空超时、子进程终止不明、显存未恢复、H3恢复响应不明。
这是有意的“失败时保持阻断”，需要运维告警和人工检查，不会冒险重启H3。

```bash
.venv/bin/python coordinator.py status
```

先检查日志和 `active.json`，确认对应Qwen进程已经结束、设备无遗留任务、审核/H3状态正常。**不要盲目删除租约文件，也不要按记录中的PID直接kill（PID可能被复用）。**

核实后可执行：

```bash
.venv/bin/python coordinator.py recover --confirm-no-qwen-process
```

恢复命令会获取相同本地锁，使用原租约重新确认暂停和排空，在有基线时检查显存，然后幂等恢复H3。若记录的未退出PID仍存在则拒绝恢复。进程启动到记录PID之间极短的崩溃窗口、主机重启、PID命名空间变化需要人工核实。

父进程被SIGKILL/断电时不能执行finally；Qwen子进程可能继续运行，但持久化H3租约必须继续阻断派发。这也是不采用自动超时解锁的原因。

## 8. 单元测试

```bash
python3 tests/test_scheduler.py
python3 -m compileall -q common.py coordinator.py worker.py install.py h3_adapter.py
```

这些测试使用模拟H3/显存数据，不下载模型、不调用NPU、不证明910B真实推理已通过。

## 9. 上游依据

- [Qwen模型与CPU卸载说明](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [QwenImage21Pipeline源码：组件卸载顺序与调用参数](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
- [Diffusers卸载API](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_utils.py)
- [Ascend PyTorch与CANN配套兼容性](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)
- [TorchNPU显存接口](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/memory.py)

参考日期：2026-09-28。公开的昇腾示例不能替代这台910B服务器的兼容性验收。
