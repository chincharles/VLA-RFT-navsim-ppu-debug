# 阿里云 ClearML：远程仓库一键启动

PPU-ZW810E 任务不要使用本页旧的 Python 3.10/CUDA 入口；使用 [PPU_ZW810E.md](PPU_ZW810E.md) 的 `debug_2ppu.sh`。普通CUDA环境才使用下文的 `clearml_start`。

**16卡完整流水线请用 [TRAIN_16GPU.md](TRAIN_16GPU.md)，Python入口是 `scripts/navsim/train_16gpu.py`。下文旧 `clearml_start` 入口是单卡最小验证。**

适用于 ClearML 已完成远程代码拉取、数据盘和持久化输出盘已挂载的任务。先把本次新增文件提交到**你配置给 ClearML 的迁移代码仓库**；只拉原始 VLA-RFT 仓库不会包含这些入口。本地没有向你的远程仓库推送，也没有提交云端任务。

## 启动命令

工作目录设为迁移仓库根目录。用户提供的服务器路径已作为默认配置写入 `configs/navsim/aliyun_paths.sh`，直接运行：

```bash
bash scripts/navsim/clearml_start.sh
```

如果界面只有 Python “Script / Entry point” 字段，填写 `scripts/navsim/clearml_start.py`，Working directory 填仓库根目录（通常为 `.`）。Python 入口会等候 Bash 子流程结束并传播退出码；平台可收集标准输出。所有路径仍可用任务环境变量覆盖。

默认输出与缓存分别为 `/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs` 和 `/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/cache`，需要任务具有写权限。每次启动会在 runs 下创建独立目录。可以先仅打印实际路径，核对平台是否遗留了其他环境变量：

```bash
bash scripts/navsim/clearml_start.sh --print-paths
```

基础镜像/运行环境必须提供 **Linux、Python 3.10（含 venv）、Bash、Git 和可用的 NVIDIA GPU 驱动接口**。入口默认使用 `python3`，可通过 `RFT_PYTHON=/实际路径/python3.10` 指定。Python 入口默认使用 ClearML 当前 Python 解释器。ClearML Agent 使用环境中已有的 Python，不会替本入口安装 Python；参见 [官方运行流程](https://clear.ml/docs/latest/docs/clearml_agent/)。

不需要在 ClearML 的环境准备区粘贴整套训练命令；该入口创建独立 venv 并调用本项目的安装脚本。任务不要预先安装原机器人项目的全套 TensorFlow/VERL 依赖。平台若自动推断任务依赖，应以这个仅含标准库的启动入口为准，训练依赖由 `configs/navsim/requirements-server.txt` 和安装脚本管理。保持 ClearML 自身正常运行所需的依赖。

## 数据与网络

默认数据根目录是 `/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1`。路径配置包括用户提供的全部七项：

```text
/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1/navsim_logs/trainval
/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1/sensor_blobs/trainval
/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1/navsim_logs/test
/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1/sensor_blobs/test
/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1/navsim_logs/mini
/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1/sensor_blobs/mini
/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1/map
```

地图使用单数 `map`，必须已经解压到实际地图根目录。`TRAIN_LOGS/TRAIN_SENSORS`、`TEST_LOGS/TEST_SENSORS`、`MINI_LOGS/MINI_SENSORS` 分别导出到环境；最小训练流程仍只使用 trainval 中的 navtrain 训练/验证划分，配置测试路径不会将测试数据加入训练。只使用固定 NAVSIM v1.1 的 PDMS。

任务需要访问 GitHub（NAVSIM/nuPlan/LPIPS）、PyPI 或你配置的包镜像、Hugging Face 和 torchvision 权重服务。只改 PyPI 镜像不能解决其他服务的网络连通性。代理、HF Token 等通过平台环境/密钥管理配置，不写进代码或启动命令，不在日志打印完整环境。

首次自动下载：

- NAVSIM 固定提交 `0811876c274e8b058ab2be9b3dcd4d37bd23f177`，放到本次输出目录；不依赖本机的兄弟目录。
- VGG16、LPIPS 权重，分块校验哈希后发布到持久化缓存。
- 默认候选 `VLA-Adapter/LIBERO-Object`，固定本次解析到的 revision，使用 HF 下载缓存。它是额外的机器人微调初始化，**不是 VLA-RFT 作者权重**；实际兼容性仍需严格加载检查。

可选环境变量：

| 变量 | 用途 |
|---|---|
| `NAVSIM_ROOT` | 已有的干净、固定提交官方 checkout；入口只检查，不 reset |
| `VLM_DIR` | 已有的兼容 HF VLM 目录；设置后跳过 VLM 下载 |
| `RFT_VLM_REPO` | 改用其他已知兼容的 HF 权重仓库 |
| `RFT_VLM_REVISION` | 指定 HF revision；实际解析出的 SHA 会记录 |

## 实际执行顺序与范围

1. 检查挂载路径、Python 版本、单进程任务配置，创建新输出目录。
2. 准备固定版本 NAVSIM、独立 venv，安装依赖并记录解析后的包版本。
3. 在平台分配的首个可见 CUDA 设备上执行真实矩阵运算，核对驱动/架构兼容性。若平台已设置 `CUDA_VISIBLE_DEVICES`，保留其首个设备标识（含 UUID/MIG），屏蔽其余设备；未设置时选择容器内首个可见设备。
4. 导出 16 个训练场景和 4 个日志隔离的验证场景；归一化只来自训练。
5. 下载/校验缺失权重，记录来源；真实 VLM 加载与前向失败立即停止，不忽略缺失 key 或维度错误。
6. tokenizer 从零训练 20 步，生成这 20 个场景的官方 metric cache。
7. WM 和 SFT 各训练 2 步并恢复追加 1 步；先评测 SFT，再进行图像奖励 RL 和官方驾驶奖励 RL 各 1 步，最后同设置评测两组策略。

这是**单卡最小真实数据流程**。不要用 torchrun 或多 worker 同时启动 bootstrap。即使任务分配 8 张卡，此入口也只使用首个可见设备进行上述验证；首次提交建议只申请 1 张。SceneLoader 会读取 trainval 日志索引，16+4 限制的是导出数量，不能据此假设 CPU 内存开销很小。

20 步 tokenizer 和 3 步 WM 不构成有效驾驶模拟器训练，不能用这些结果作正式性能结论。完整训练及 8/16/32 卡阶段应在最小流程通过后单独配置。96GB 是容量，实际 GPU 架构仍必须受本项目固定 torch/CUDA 支持。

## 日志、缓存与重新提交

每次自动新建 `$RFT_OUTPUT_ROOT/run-UTC时间-随机后缀/`，控制台首先输出 `Run directory:`。其中包含：

```text
launcher.log                    全部子阶段 stdout/stderr
launcher-status.json            正常退出或失败时的退出码与最后阶段
requirements-resolved.txt       实际 pip 解析版本
environment.json                GPU、软件版本、训练预算、NAVSIM/code commit
weights.json                    下载/本地权重来源和实际路径
vlm-check.json                  真实样本 VLM 检查（成功或失败）
venv/                          本次独立训练环境
data/                          16+4 导出及训练统计
config/                        实际训练配置
tokenizer/                     tokenizer checkpoint 与 pretrained 目录
metric-cache/                  官方训练/验证 cache
run/wm/                        世界模型 checkpoint、逐步日志
run/sft/                       SFT checkpoint、逐步日志
run/rl_img/                    图像奖励 RL
run/rl_drive/                  驾驶奖励 RL
run/eval_sft/                   官方 PDMS 输出
run/eval_rl_img/
run/eval_rl_drive/
```

下载缓存跨任务复用，但每次重新提交默认是**新训练**，不自动续训旧任务。缓存目录需使用支持 POSIX 文件锁、硬链接的本地盘或 NAS，不能直接使用 OSS 对象存储 FUSE 挂载。已有缓存损坏时报错并保留文件，不静默覆盖；中断的临时下载不会作为有效权重。正式续训须使用已有训练入口的 `--resume` 并保持配置、随机种子和 world size 相同，见主 README。

已完成本地启动脚本检查；未在阿里云 ClearML 提交运行，未验证云端依赖安装、预训练 VLM 兼容性或真实训练/评测成绩。
