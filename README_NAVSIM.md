# VLA-RFT → NAVSIM v1.1：非官方迁移实现

本目录是在用户提供的源码快照上创建的独立 Git 仓库，分支 `navsim/v1.1-migration`。原目录 `VLA-RFT-main` 未修改。

**服务器只有 NAVSIM 数据、尚无其他权重时，先按 [START_SERVER.md](START_SERVER.md) 执行。** 其中包含预训练奖励权重下载、tokenizer 从零启动、候选 VLM 的实际加载检查，以及只计算所选场景的 metric cache 命令。

**阿里云 ClearML 从远程仓库启动：** 见 [CLEARML.md](docs/navsim/CLEARML.md)，配置四个挂载路径后执行 `bash scripts/navsim/clearml_start.sh`；也提供 Python entry point。

**状态：已有实际训练、rollout、奖励、checkpoint、官方 evaluator 接入代码；本地已执行合成输入的真实模型计算与恢复测试。尚未执行真实 NAVSIM 数据训练、预训练 VLM 推理、预训练 LPIPS 奖励或官方 PDMS。不得将本地 smoke 称为 NAVSIM 复现成绩。**

固定官方标签提交 `0811876c274e8b058ab2be9b3dcd4d37bd23f177`，4 s / 0.5 s / 8 poses，PDMS。不要 checkout 同名 v1.1 分支：其提交不同。不要使用 v2 EPDMS。

- 论文与源码审计：[DIFFERENCES.md](docs/navsim/DIFFERENCES.md)
- 已执行与未验证记录：[EXPERIMENTS.md](docs/navsim/EXPERIMENTS.md)
- 缺失资源：[RESOURCES.md](docs/navsim/RESOURCES.md)

## 1. 上传与服务器环境

上传整个本目录；本地 `.venv-navsim` 不需要上传。建议 Linux、Python 3.10、CUDA 12.x、独立虚拟环境。以下命令从本目录执行。安装脚本是待服务器验证的依赖方案，**不是已经验证的服务器环境锁文件**。

```bash
git clone https://github.com/autonomousvision/navsim.git /data/code/navsim
git -C /data/code/navsim checkout --detach 0811876c274e8b058ab2be9b3dcd4d37bd23f177
python3.10 -m venv .venv-server
source .venv-server/bin/activate
export NAVSIM_ROOT=/data/code/navsim
bash scripts/navsim/install_server.sh
source scripts/navsim/env.sh
export NUPLAN_MAPS_ROOT=/data/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/openscene
export NAVSIM_EXP_ROOT=/data/experiments/navsim-rft
```

安装脚本保留 NAVSIM 仿真依赖，对 torch/torchvision/Hydra 采用显式兼容版本；不安装原 robot RLDS/TensorFlow/VERL 训练栈。这里的新训练入口复用原架构，但不是原 Ray/FSDP 引擎。没有模型下载或自动大作业。

准备本地 Prismatic/OpenVLA-compatible VLM 目录（config、processor、tokenizer、所有权重分片）；优先原实现使用的 hidden=896 小型 VLM。机器人 7D 动作头不直接加载到驾驶 3D 动作头。VLM、视觉 projector 可迁移；驾驶 action projector、proprio projector、DiT/Sigma 重新训练。完整加载迁移 checkpoint 时严格匹配，不静默 `strict=False` 跳过 action 维度。

原 Compressive FSQ tokenizer 目录需要 `config.json` 和 diffusion 权重，输入 256×256、context_length=1、32×32 context tokens、8×8 dynamic tokens。世界模型 backbone 可从零训练，也可指定兼容 Llama 目录，但机器人世界模型必须重新用驾驶数据训练。

图像奖励必须提供两个实际预训练文件：

```bash
export VLA_RFT_VGG16_PATH=/data/weights/vgg16-397923af.pth
export VLA_RFT_LPIPS_PATH=/data/weights/vgg.pth
```

LPIPS 使用发布源码 VGG 版本，linear 权重 MD5 必须为 `d507d7349b931f0638a25a48a722f98a`。原源码缺 VGG 文件时会退回随机网络，此行为已禁止。

## 2. 数据检查与预处理

```bash
export TRAIN_LOGS=$OPENSCENE_DATA_ROOT/navsim_logs/trainval
export TRAIN_SENSORS=$OPENSCENE_DATA_ROOT/sensor_blobs/trainval
python -m navsim_rft.cli export --root "$NAVSIM_ROOT" \
  --logs "$TRAIN_LOGS" --sensors "$TRAIN_SENSORS" \
  --split navtrain --role train --limit 16 --out /data/rft-cache/train16
python -m navsim_rft.cli export --root "$NAVSIM_ROOT" \
  --logs "$TRAIN_LOGS" --sensors "$TRAIN_SENSORS" \
  --split navtrain --role val --limit 4 --out /data/rft-cache/val4 \
  --exclude /data/rft-cache/train16/manifest.json
```

- 只从 navtrain 的按日志 SHA256 留出之外的训练日志取训练样本；10% 日志用于 validation。训练与验证不用同一日志的相邻重叠 clip。
- manifest 保存 evaluator token（initial_token）、scene_token、log_name、时间戳、相机时间戳证据、合法文本导航输入。
- 使用官方 `get_future_trajectory(8)` 后轴局部位姿；前视图像由同一官方 frame 读取。检查相邻 frame 0.5s ±50ms。若原相机字典没有独立 timestamp，仅记录官方 frame association，**不声称独立传感器同步已验证**。
- 输出初始原分辨率图像、4 帧历史、8 帧未来 256px 图像、ego state、专家轨迹、mask。第一版要求完整有效未来帧，缺帧立即报错，不把黑帧当奖励监督。
- `stats.json` 只在 train 导出时生成；记录 token 列表和 hash。训练拒绝非训练 manifest 和统计来源不匹配。策略预测 SE(2) 局部增量，积分成官方初始车体系位姿。
- 使用前视一视角。manifest 有 views 字段，数据保留历史图像；当前 encoder 只用最新图像，多视角/历史融合尚未实现，不能直接改配置声称支持。
- 导出目录必须新建，避免覆盖原数据。完整导出会消耗磁盘，先用小 limit。

## 3. Metric cache

先生成 **训练/验证** cache，供 B 的 RL 与 validation 共用；测试 cache 仅供最终评测：

```bash
python "$NAVSIM_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtrain cache.cache_path=/data/rft-metric-cache/train \
  navsim_log_path="$TRAIN_LOGS" \
  worker=sequential
```

metric cache 的 metadata CSV 常包含绝对路径，上传/迁移后需重新生成或合法修正路径。B 逐 token 校验训练白名单；评测拒绝缺失 cache，避免官方 evaluator 静默在子集上给分。

## 4. 配置与预检

```bash
python scripts/navsim/configure.py \
  --navsim-root "$NAVSIM_ROOT" --train-manifest /data/rft-cache/train16/manifest.json \
  --stats /data/rft-cache/train16/stats.json --vlm /data/weights/vla-base \
  --tokenizer /data/weights/video-tokenizer --metric-cache /data/rft-metric-cache/train \
  --output /data/rft-configs/smoke
python -m navsim_rft.preflight --config /data/rft-configs/smoke/rl_img.json \
  --output /data/experiments/preflight.json
```

生成 `wm.json`、`sft.json`、`rl_img.json`（A）、`rl_drive.json`（B），以及独立的 log-future 参考和 source-math 对照配置。A 默认保留发布启动脚本的 expert-rollout 参考，公式归约按 paper；差异表明确记载这种选择。`sft_source_math.json` 必须先训练，source模式RL不能加载paper模式SFT（flow符号不同，会报错）。`rl_img_source_math` 仅是发布源码数学差异对照，不能当成逐比特原实验复现。

预检实际 import 依赖、核对官方 checkout、检查权重分片、读取一个数据样本和 GPU 资源。缺失就报错，不自动换成玩具模型。

## 5. 最小真实数据链路（单卡，明确步数）

```bash
export CONFIG_DIR=/data/rft-configs/smoke
export RUN_DIR=/data/experiments/rft-smoke-001
export VAL_MANIFEST=/data/rft-cache/val4/manifest.json
export ACTION_STATS=/data/rft-cache/train16/stats.json
export TRAIN_METRIC_CACHE=/data/rft-metric-cache/train
bash scripts/navsim/minimal_server.sh
```

顺序：WM 2 步 → 恢复 1 步 → 单步/连续视频诊断 → SFT 2 步 → 恢复 1 步 → 官方基线评测 → A/B 各 1 步 → 同设置官方评测。少量步数只验证流水线，不能说明驾驶世界模型有效或策略性能有提升。

分开运行训练及恢复：

```bash
python -m navsim_rft.train --config "$CONFIG_DIR/wm.json" --output /data/experiments/wm --steps 20
python -m navsim_rft.train --config "$CONFIG_DIR/sft.json" --output /data/experiments/sft --steps 20
python -m navsim_rft.train --config "$CONFIG_DIR/sft.json" --output /data/experiments/sft \
  --resume /data/experiments/sft/step-000020.pt --steps 5
python -m navsim_rft.train --config "$CONFIG_DIR/rl_img.json" --output /data/experiments/image-rl --steps 5 \
  --init-policy /data/experiments/sft/step-000025.pt --wm-checkpoint /data/experiments/wm/step-000020.pt
python -m navsim_rft.train --config "$CONFIG_DIR/rl_drive.json" --output /data/experiments/drive-rl --steps 5 \
  --init-policy /data/experiments/sft/step-000025.pt
```

`--steps` 是本次追加的 optimizer steps。checkpoint 包含模块、AdamW、每 rank Python/NumPy/Torch/CUDA RNG、step、配置、统计和版本；只允许同 world size/config/seed 恢复。原 VERL 分片 ckpt 不是这个格式，不可作为 `--resume`。`--init-policy` 必须是本项目 SFT ckpt。每次运行终点保存一次；长任务应分段执行并 resume，当前不提供崩溃前的自动中途周期存档。

如果缺 tokenizer 权重，提供独立的**重实现**训练入口：

```bash
python -m navsim_rft.train_tokenizer --manifest /data/rft-cache/train16/manifest.json \
  --stats /data/rft-cache/train16/stats.json --output /data/experiments/tokenizer --steps 20
# 新 tokenizer 路径：/data/experiments/tokenizer/pretrained-000020
```

该入口复用原 FSQ/conditional encoder-decoder，以 L1+LPIPS 重建训练，不是作者未发布的 tokenizer 预训练 recipe；不能用20步模型当合格模拟器。支持 `--init` 和 `--resume`。缺 VLM 权重没有伪造替代，需提供兼容预训练模型。

## 6. 官方评测与三组比较

```bash
python -m navsim_rft.evaluate --navsim-root "$NAVSIM_ROOT" \
  --checkpoint /data/experiments/sft/step-000025.pt \
  --logs "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensors "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --metric-cache /data/rft-metric-cache/test --split navtest \
  --output /data/experiments/eval-sft
```

A/B 替换 checkpoint、独立 output，保持 split/seed/cache 完全一致。推理只接收 `AgentInput`，固定同一个初始 Gaussian noise 用 ODE 生成一条轨迹；不访问 future、Scene 或 reward，不用测试真值挑候选。这是明确的推理协议选择，见差异表。

包装器调用未经修改的官方 `run_pdm_score.py`，runtime Hydra agent 配置写在输出目录，不改 NAVSIM 仓库。CSV 保存 token、valid、官方 total/分项与 average；如有失败或 token 遗漏会报错。未在真实服务器上执行此集成，必须先跑 baseline 验证。

```bash
python -m navsim_rft.report --sft /path/sft.csv --image /path/image.csv \
  --driving /path/driving.csv --output /data/experiments/comparison
python -m navsim_rft.quality --checkpoint /data/experiments/wm/step-000020.pt \
  --manifest "$VAL_MANIFEST" --stats "$ACTION_STATS" --output /data/experiments/wm-quality --limit 4
python -m navsim_rft.analyze_rewards --policy /data/experiments/sft/step-000025.pt \
  --world /data/experiments/wm/step-000020.pt --manifest /data/rft-cache/train16/manifest.json \
  --stats "$ACTION_STATS" --metric-cache "$TRAIN_METRIC_CACHE" \
  --output /data/experiments/reward-analysis --limit 4 --candidates 4
```

质量 GIF 从左到右是真值、teacher-forced 单步预测、连续 rollout，JSON 含逐时刻 L1/LPIPS/PSNR。相关性命令仅用训练场景，保存两个图像参考奖励、PDMS、Pearson/Spearman、排名不一致案例和候选轨迹，不用于测试候选选择。

## 7. 8 / 16 / 32 张 96GB 卡

先单卡小流程，再单机8卡测试通信和恢复，最后扩到多机。新 runner 是每卡完整模型的同步数据并行，**不降低单模型显存**；没有声称原 FSDP/vLLM 部署已迁移或多节点性能已验证。小型 VLM 是合理起点；7B 全参 FP32 Adam 的参数/梯度/状态约112GB，尚未计激活，不适合本入口每卡96GB。

```bash
export CONFIG=/data/rft-configs/smoke/sft.json
export OUTPUT=/data/experiments/sft-8gpu
export STEPS=100
export NPROC_PER_NODE=8
bash scripts/navsim/distributed.sh
# 16卡：2节点，每节点8卡；32卡：4节点，每节点8卡。
# 每节点设置同一 MASTER_ADDR、NNODES，各自 NODE_RANK；共享数据/输出/checkpoint路径。
```

默认每卡1个场景、N=4。8/16/32卡每步最多8/16/32场景与32/64/128候选；数据很小时跨 rank 可能重复，日志列出实际 token，不能把候选数当独立场景数。RL 的 VLM 和 WM 冻结，flow head、两个 projector、Sigma 更新；SFT 默认更新 VLM/query/flow/projectors，Sigma不更新；WM只更新Llama，tokenizer冻结。

没有 GPU 型号、互连、权重规模和真实步时，不能给可信的固定小时数。**估算方式**：先执行10–20步，用 `metrics.rank*.jsonl` 的稳态 `seconds` 中位数 × 目标步数，加上 cache/预处理/验证时间；每阶段独立计时。32卡不保证比8卡快4倍。逐rank日志有峰值显存、token、loss、奖励、优势、梯度范数、概率重算误差、实际步数与seed；首次参数哈希校验也有额外CPU/IO开销。

## 8. 本地可复查测试

```bash
PYTHONPATH="$PWD" python -m unittest discover -s tests/navsim -v
PYTHONPATH="$PWD:$PWD/train/verl" python tests/navsim/smoke_world.py --output /tmp/rft-synthetic-new
```

后者明确标注 SYNTHETIC，随机 tokenizer + 小 Llama + 原 DiT 变体，L1-only smoke，不是 A 的 LPIPS 实验。不把它的 loss/reward 解释为驾驶质量。检查结果位于 `outputs/local-checks/`。
