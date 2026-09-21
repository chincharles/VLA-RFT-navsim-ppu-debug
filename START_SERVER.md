# 只有 NAVSIM 数据时，从零启动

以下是服务器执行命令，不代表已经在服务器验证成功。固定 NAVSIM v1.1 标签提交，使用 PDMS。先单卡：16 条训练、4 条独立日志验证；tokenizer 20 步、WM/SFT 各 3 步、两种 RL 各 1 步。这个流程只验证接口和训练链路，不检验收敛或复现成绩。不要把该小模型用于正式结论。

在同一个 **Bash** 会话中按顺序执行。先上传并解压交付包，其内同时包含 `VLA-RFT-navsim`、`NAVSIM-v1.1`。不要把已有数据移入代码目录。新输出目录不能与旧实验重名。完整解释见 `README_NAVSIM.md`。

## 1. 设置路径

前五个路径改成服务器实际位置。`RFT_WORK` 必须是新实验目录。数据下载了 v1/v2 多套时，这里选择 v1 的 trainval logs/blobs；不使用 v2 配置。

```bash
export PROJECT_ROOT=/data/work/VLA-RFT-navsim
export NAVSIM_ROOT=/data/work/NAVSIM-v1.1
export OPENSCENE_DATA_ROOT=/data/openscene
export NUPLAN_MAPS_ROOT=/data/maps
export RFT_WORK=/data/experiments/rft-start-001

cd "$PROJECT_ROOT"
set -euo pipefail
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT="$RFT_WORK/navsim-exp"
export TRAIN_LOGS="$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
export TRAIN_SENSORS="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"
export TRAIN_MANIFEST="$RFT_WORK/data/train16/manifest.json"
export VAL_MANIFEST="$RFT_WORK/data/val4/manifest.json"
export ACTION_STATS="$RFT_WORK/data/train16/stats.json"
export TRAIN_METRIC_CACHE="$RFT_WORK/metric-cache"
export CONFIG_DIR="$RFT_WORK/config"
export RUN_DIR="$RFT_WORK/run"
export VLM_DIR="$RFT_WORK/weights/vla-adapter-object"
export VLA_RFT_VGG16_PATH="$RFT_WORK/weights/vgg16-397923af.pth"
export VLA_RFT_LPIPS_PATH="$RFT_WORK/weights/vgg.pth"
mkdir -p "$RFT_WORK/weights" "$RFT_WORK/logs" "$NAVSIM_EXP_ROOT"
test -d "$TRAIN_LOGS"
test -d "$TRAIN_SENSORS"
test -d "$NUPLAN_MAPS_ROOT"
test "$(git -C "$NAVSIM_ROOT" rev-parse HEAD)" = 0811876c274e8b058ab2be9b3dcd4d37bd23f177
```

maps 根目录应包含 nuPlan 地图版本文件与城市地图文件夹，不能只包含 sensor_blobs。官方导出器第一次构造 Scene 就访问地图。所有以上环境变量在新终端里需要重新设置。

如果只下载了 logs/blobs、没有 maps，先补地图。下面 URL 来自随包固定版本的 `NAVSIM-v1.1/download/download_maps.sh`，压缩包名为 v1.1、里面的地图目录为 v1.0，不是 NAVSIM v2 数据：

```bash
mkdir -p "$RFT_WORK/map-download"
curl -fL --retry 3 \
  https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip \
  -o "$RFT_WORK/map-download/nuplan-maps-v1.1.zip"
unzip -n "$RFT_WORK/map-download/nuplan-maps-v1.1.zip" -d "$RFT_WORK/map-download"
export NUPLAN_MAPS_ROOT="$RFT_WORK/map-download/nuplan-maps-v1.0"
test -d "$NUPLAN_MAPS_ROOT"
```

## 2. 安装并检查 CUDA

已有 Python 3.10 即可；以下使用独立 venv，安装脚本不接受未激活 venv 的系统环境。

```bash
python3.10 -m venv .venv-server
source .venv-server/bin/activate
bash scripts/navsim/install_server.sh 2>&1 | tee "$RFT_WORK/logs/install.log"
source scripts/navsim/env.sh
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
x = torch.randn(512, 512, device='cuda')
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('CUDA computation OK')
PY
```

96GB 是显存大小，不是显卡型号。若实际卡需要比当前 torch 2.4.1 更新的 CUDA 架构支持，在这一步停止并保留报错，不要跳过 CUDA 验证继续训练。当前固定依赖尚未在你的服务器安装验证。

## 3. 读取真实数据并导出小样本

```bash
python -m navsim_rft.cli export --root "$NAVSIM_ROOT" \
  --logs "$TRAIN_LOGS" --sensors "$TRAIN_SENSORS" \
  --split navtrain --role train --limit 16 --out "$RFT_WORK/data/train16"
python -m navsim_rft.cli export --root "$NAVSIM_ROOT" \
  --logs "$TRAIN_LOGS" --sensors "$TRAIN_SENSORS" \
  --split navtrain --role val --limit 4 --out "$RFT_WORK/data/val4" \
  --exclude "$TRAIN_MANIFEST"
```

这一步实际检查 frame 时间、图像与轨迹，生成训练集专用统计。训练/验证按日志划分，未来图像只作为训练目标或奖励参考。

## 4. 下载 VGG16 与 LPIPS

VGG 来源为 torchvision 官方权重服务；LPIPS 来源为作者 PerceptualSimilarity 仓库。已在本地核对下面 LPIPS 文件 MD5 与 VLA-RFT 源码要求一致。

```bash
curl -fL --retry 3 https://download.pytorch.org/models/vgg16-397923af.pth \
  -o "$VLA_RFT_VGG16_PATH"
curl -fL --retry 3 https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth \
  -o "$VLA_RFT_LPIPS_PATH"
python - <<'PY'
import hashlib, os
from pathlib import Path
vgg = Path(os.environ['VLA_RFT_VGG16_PATH'])
lpips = Path(os.environ['VLA_RFT_LPIPS_PATH'])
assert hashlib.sha256(vgg.read_bytes()).hexdigest().startswith('397923af'), 'VGG hash mismatch'
assert hashlib.md5(lpips.read_bytes()).hexdigest() == 'd507d7349b931f0638a25a48a722f98a', 'LPIPS hash mismatch'
print('Reward weights OK')
PY
```

## 5. 从零训练图像 tokenizer，先跑 20 步

```bash
python -m navsim_rft.train_tokenizer \
  --manifest "$TRAIN_MANIFEST" --stats "$ACTION_STATS" \
  --output "$RFT_WORK/tokenizer" --steps 20 --seed 42 \
  2>&1 | tee "$RFT_WORK/logs/tokenizer.log"
export TOKENIZER_DIR="$RFT_WORK/tokenizer/pretrained-000020"
```

该入口是复用原 FSQ 网络的预训练**重实现**，不是作者已发布的预训练权重或完整训练 recipe；20 步远不足以得到有效驾驶 tokenizer。world model 不另需下载预训练 Llama，默认从零训练。

## 6. 下载小型 VLM 候选并生成配置

采用公开的 [VLA-Adapter/LIBERO-Object](https://huggingface.co/VLA-Adapter/LIBERO-Object/tree/main) 作为**待加载验证的初始化候选**。其模型卡声明 Prismatic + Qwen2.5-0.5B，HF 文件树带完整 processor/tokenizer/模型权重。这不是 VLA-RFT 作者权重；也不能称为完全匹配的原论文初始化。本地没有下载大模型或验证它的实际权重兼容性，下一步必须通过严格加载与真实数据前向。

不下载它额外的机器人 action_head/proprio `.pt`，驾驶动作头由本项目重新初始化。下面固定本次解析到的仓库 revision，并保存来源记录。

```bash
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = 'VLA-Adapter/LIBERO-Object'
revision = HfApi().model_info(repo).sha
snapshot_download(repo_id=repo, revision=revision, local_dir=os.environ['VLM_DIR'],
    allow_patterns=['*.json', '*.txt', '*.safetensors', 'tokenizer.model'])
Path(os.environ['RFT_WORK'], 'weights', 'vlm_source.json').write_text(json.dumps(
    {'repo': repo, 'revision': revision, 'status': 'candidate; strict load required',
     'difference': 'robot-finetuned VLA-Adapter backbone, not released VLA-RFT weights'}, indent=2))
PY
python scripts/navsim/configure.py \
  --navsim-root "$NAVSIM_ROOT" --train-manifest "$TRAIN_MANIFEST" \
  --stats "$ACTION_STATS" --vlm "$VLM_DIR" --tokenizer "$TOKENIZER_DIR" \
  --metric-cache "$TRAIN_METRIC_CACHE" --output "$CONFIG_DIR"
python scripts/navsim/check_vlm.py \
  --config "$CONFIG_DIR/sft.json" --output "$RFT_WORK/vlm-check.json"
```

如果这一步因缺失 key、维度不兼容或 processor 报错，不要改为忽略权重错误。保留 `vlm-check.json` 和 traceback；需要兼容 checkpoint 或明确的转换适配。VLM 失败时，已生成配置下的 WM 训练仍可独立运行，它不读取 VLM；SFT/RL/策略评测必须等 VLM 检查通过。

## 7. 生成这 20 个场景的官方 metric cache

```bash
python scripts/navsim/cache_subset.py --navsim-root "$NAVSIM_ROOT" \
  --logs "$TRAIN_LOGS" --manifests "$TRAIN_MANIFEST" "$VAL_MANIFEST" \
  --cache "$TRAIN_METRIC_CACHE" \
  2>&1 | tee "$RFT_WORK/logs/metric-cache.log"
```

helper 将 manifest 的实际 token 合集传给官方脚本，不启动全量 cache。输出仍是官方 NAVSIM cache 格式。不要将 NAVSIM v2 cache 传给这个 v1.1 流程。

## 8. 最小真实训练、恢复、两种 RL、官方评测

```bash
bash scripts/navsim/minimal_server.sh 2>&1 | tee "$RFT_WORK/logs/minimal.log"
```

脚本按顺序执行（任一步失败即停止）：

1. 依赖、权重、数据预检。
2. WM 训练 2 步，checkpoint 恢复再训练 1 步；保存单步/连续 rollout 对比。
3. SFT 训练 2 步，恢复再训练 1 步。
4. 在同一组 4 个 held-out navtrain 场景上执行官方 SFT baseline PDMS。
5. A：图像奖励 RL 1 步；B：官方驾驶奖励 RL 1 步，两组分别从同一 SFT 初始化。
6. 在相同的 4 个验证场景上分别评测 A/B。

默认 A 使用源启动配置的 expert-action rollout 作为参考图像，paper 数学模式；log-future/source-math 是分开的配置，详见差异表。这里是一次动作块采样后模拟整个未来，不是长时程闭环策略交互。

主要产物：

```text
$RFT_WORK/tokenizer/pretrained-000020/
$RUN_DIR/preflight.json
$RUN_DIR/wm/step-000003.pt
$RUN_DIR/wm_quality/
$RUN_DIR/sft/step-000003.pt
$RUN_DIR/rl_img/step-000001.pt
$RUN_DIR/rl_drive/step-000001.pt
$RUN_DIR/eval_sft/
$RUN_DIR/eval_rl_img/
$RUN_DIR/eval_rl_drive/
```

各训练目录含 `metrics.rank0.jsonl`。官方 CSV 位于各 eval 目录内；失败场景不能被当作成功评测忽略。

## 9. 最小流程成功后，8 卡通信验证

以下仅追加一个 20 步的 SFT 数据并行检查；仍用 train16，不能当正式训练。每卡保留完整模型，不是 FSDP，不会把单卡模型显存除以 8。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export CONFIG="$CONFIG_DIR/sft.json"
export OUTPUT="$RFT_WORK/sft-8gpu-check"
export STEPS=20
bash scripts/navsim/distributed.sh 2>&1 | tee "$RFT_WORK/logs/sft-8gpu.log"
```

恢复这个 8 卡任务，必须保持配置/卡数/seed 不变：

```bash
export STEPS=5
bash scripts/navsim/distributed.sh --resume "$OUTPUT/step-000020.pt"
```

`--steps` 为本次追加步数，上例结果为 `step-000025.pt`。单卡 checkpoint 不能用 `--resume` 转 8 卡。tokenizer 入口当前仅单进程，不要使用 torchrun 多进程同时写同一 tokenizer 输出目录。

## 10. 从小验证转正式训练

先扩大 **navtrain** 导出范围，建立新数据目录、训练统计、新配置与新实验；同一正式实验三组用相同训练统计和验证集合。不要复用 train16 的统计或把测试集加入导出、奖励、调参。先让 tokenizer 重建质量和 WM 单步/连续生成在 held-out navtrain 上达到可用水平，再进行 RL；不能用 20 步 tokenizer + 3 步 WM 开始宣称有效强化学习。

按每阶段的真实稳态步时估算预算，分段运行。当前训练只在一次命令的末尾保存 checkpoint，长任务应按可承受损失的步数分段并恢复。16/32 卡分别设置 `NNODES=2/4`、每机 `NPROC_PER_NODE=8`、各机不同 `NODE_RANK`、同一个主节点 IP 和共享路径；它们尚未在真实服务器验证。不要先自动启动大规模任务。

逐阶段的独立命令、正式官方测试集评测、三组比较和奖励相关性分析见 `README_NAVSIM.md`；最终报告要区分实现、服务器实际执行、未验证部分。
