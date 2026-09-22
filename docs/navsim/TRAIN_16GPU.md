# 阿里云 ClearML：16卡完整启动

入口已写入用户的CPFS路径。脚本在用户运行后自动安装依赖、下载权重、生成训练数据和官方cache，顺序执行tokenizer → WM → SFT → 基线评测 → 图像奖励RL A → 评测 → 驾驶奖励RL B → 评测与比较。A/B都从相同SFT出发。

## 1. ClearML任务配置

- 代码：上传本代码包至你的远程Git仓库，ClearML拉取该仓库。工作目录为仓库根目录。
- Python入口：`scripts/navsim/train_16gpu.py`。平台支持Shell命令时也可用下方bash入口。
- 容器：Linux、Python **3.10**，具备`git`、`python3 -m venv`及可用NVIDIA驱动/GPU；能访问PyPI、GitHub、Hugging Face和PyTorch权重站点。
- 单机分配16张GPU；双机每台分配8张GPU。脚本每台机器启动一次，不是每张卡启动一次。
- CPFS以相同绝对路径挂载到每台机器。双机必须能互相连接master端口和NCCL通信端口；网络接口参数使用平台配置，可通过`NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`等环境变量传入。

该入口在任务容器内部调用torchrun，不负责申请GPU、提交第二台机器或调用ClearML SDK启动远程任务。双机需要平台实际启动两个节点，并为各自设置下方参数。Python入口与bash入口接受相同参数。

## 2. 单机16卡

在仓库根目录运行这一个命令：

```bash
bash scripts/navsim/train_16gpu.sh
```

ClearML只接受Python入口时，入口填写`scripts/navsim/train_16gpu.py`，脚本参数留空。默认使用当前Python3.10解释器，创建各节点独立虚拟环境。

## 3. 双机各8卡

把下方`10.0.0.10`替换为**节点0的可达内网IP**。两台使用相同代码、相同唯一run-id、相同master地址和端口。每次新训练换一个run-id。

节点0：

```bash
NNODES=2 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
RFT_RUN_ID=navsim16-run001 bash scripts/navsim/train_16gpu.sh
```

节点1：

```bash
NNODES=2 NPROC_PER_NODE=8 NODE_RANK=1 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
RFT_RUN_ID=navsim16-run001 bash scripts/navsim/train_16gpu.sh
```

也可以在ClearML各节点环境变量中填写这些值，Python入口无需参数。不要把上面两条都放在同一节点顺序执行。

## 4. 固定数据路径与自动下载

默认路径来自`configs/navsim/aliyun_paths.sh`：

| 用途 | 路径 |
|---|---|
| 数据根 | `/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1` |
| 训练/验证日志 | 同上根目录下`navsim_logs/trainval` |
| 训练/验证传感器 | 同上根目录下`sensor_blobs/trainval` |
| 地图 | 同上根目录下`map` |
| 新训练输出 | `/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs/<run-id>` |
| 下载缓存 | `/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/cache` |

下载/初始化内容：

| 模块 | 行为 |
|---|---|
| NAVSIM源码 | 拉取固定v1.1提交`0811876c274e8b058ab2be9b3dcd4d37bd23f177` |
| VLM | 下载候选`VLA-Adapter/LIBERO-Object`的HF文件，记录解析后的revision |
| 图像奖励 | 下载预训练VGG16及LPIPS VGG线性权重，校验哈希 |
| 图像tokenizer | 没有可确认的作者权重，复用原FSQ架构从训练图像训练；这是补充重实现 |
| 世界模型、驾驶动作头、Sigma | 按迁移架构初始化并由本流水线训练 |

**候选VLM尚未在本地以真实权重验证。** 入口严格检查权重加载与一张真实训练图像的特征；不匹配会报错停止，不用随机VLM替代。已有兼容模型可设置`VLM_DIR=/绝对路径`；更换下载源可设置`RFT_VLM_REPO`和`RFT_VLM_REVISION`，但必须仍兼容本项目Prismatic实现。

只训练navtrain中按日志划分的训练部分；10%日志留出。默认选128个留出场景做三组独立PDMS评测。test/navtest不参与训练、统计、RL奖励或候选挑选。本入口没有自动最终test评测。

## 5. 默认训练预算和小样本试跑

`configs/navsim/train16.json`可在新任务开始前修改：

| 阶段 | 全局optimizer步数 |
|---|---:|
| Tokenizer补充训练 | 10,000 |
| 世界模型MLE | 20,000 |
| 策略监督预训练 | 10,000 |
| 图像奖励RL A | 1,000 |
| 官方驾驶奖励RL B | 1,000 |

每卡1个场景，全局16场景；RL每场景4候选。seed42，每500步保存，末步也保存。tokenizer入口固定每卡1场景、每步随机一个未来帧重建；JSON的`batch_size_per_gpu`用于WM/SFT/RL。train_limit=0表示导出完整训练部分，不是0样本。

这些步数是可控的初始预算，不是论文训练配方，也不保证世界模型有效或策略提升。单步/连续视频诊断会保存，但目前没有经过校准的质量阈值自动阻止低质量WM进入RL；必须结合质量结果解读A。

建议首次在同一16卡配置试跑小预算，仍会走实际模型下载、真实数据训练和官方评测：

```bash
bash scripts/navsim/train_16gpu.sh --profile smoke
```

smoke覆盖32训练/8验证、tokenizer/WM/SFT各2步、A/B各1步。双机时保留第3节环境变量，两节点都追加`--profile smoke`。这只验证链路，不是训练成绩。只看预算和路径、不执行任务：

```bash
bash scripts/navsim/train_16gpu.sh --print-plan
```

每卡保存完整FP32模型和优化器，本入口没有FSDP分片；16卡不会把单卡模型显存除以16。默认小型VLM，勿直接换成7B全参模型。启动先做每GPU矩阵运算与NCCL all-reduce检查；若实际GPU架构/驱动不兼容固定Torch2.4.1，会停止，需要针对实际型号调整依赖并重新验证。

## 6. 中断恢复

不要改源码、JSON预算、seed、节点数量或数据路径。单机：

```bash
bash scripts/navsim/train_16gpu.sh \
  --resume-run /mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs/你的run-id
```

双机保留各自NNODES/NPROC_PER_NODE/NODE_RANK/MASTER_ADDR设置，两节点均追加：

```bash
--resume-run /mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs/navsim16-run001 --attempt-id retry001
```

每次重启更换相同的新attempt-id。恢复smoke时还必须保留`--profile smoke`。脚本选择各阶段最新完整checkpoint，补足原预算；已完成的训练跳过，评测写入新的attempt目录。部分导出目录会改名保留后重做；不会覆盖旧checkpoint。旧版本单卡tokenizer checkpoint缺少新元数据，不能用于这里的resume。

已保存checkpoint含模型、优化器、每rank随机状态。中断后最多重算保存间隔内的步骤；不声称中断前未保存的步骤可恢复。所有中间checkpoint默认保留，需为模型/优化器、图像导出和metric cache预留足够CPFS空间。

## 7. 输出和验证状态

每次任务会打印run目录。其下：

- `weights.json`：权重路径、来源、版本和校验记录。
- `data/train`、`data/val`、`metric-cache`：训练导出、统计和官方缓存。
- `tokenizer`、`wm`、`sft`、`rl_img`、`rl_drive`：checkpoint、各rank逐步JSONL指标。
- `launches/<attempt>/node-N`：分阶段完整日志、依赖版本清单。
- `launches/<attempt>/nccl`：16卡实际设备、矩阵运算/通信检查结果。
- `launches/<attempt>/wm-quality`：单步/连续图像质量及GIF。
- `launches/<attempt>/eval_*`、`comparison`、`reward-analysis`：官方逐场景结果、三组对比和奖励相关性。
- 出错时`launches/<attempt>/FAILED.node-N.json`标识阶段；具体异常查看该节点日志。

本地已完成CPU双进程梯度/恢复和Hydra配置合成验证，未连接阿里云、未执行真实16卡NCCL或NAVSIM训练评测。服务器依赖安装、候选VLM兼容性、显存和耗时仍需实测；此项目是非官方迁移实现。
