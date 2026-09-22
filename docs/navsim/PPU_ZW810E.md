# ZW810E：按用户实际镜像适配的两卡入口

## 目标环境

来自用户2026-09-21提供的环境报告：Ubuntu24.04.2、Python3.12.3、Torch2.6.0、TorchVision0.21.0、TorchAudio2.6.0、Triton3.1.0；两张PPU-ZW810E，每卡95.62GiB。torch.cuda.is_available()为True，torch编译版本报告12.8。PPU-SMI1.22，报告实际显示驱动1.4.1-816bc0、HGGC13.0（此前口述为1.4.4，以最新完整报告为准）。

HGGC版本不是安装普通NVIDIA CUDA wheel的依据。运行时继续使用厂商兼容的torch.cuda接口和NCCL兼容通信入口；没有改成未确认存在的torch.ppu，也没有安装新驱动、SDK或通信库。

参考：[阿里云PPU SDK说明](https://help.aliyun.com/zh/document_detail/3011255.html)、[Torch-FL的PPU接口/通信说明](https://github.com/flagos-ai/Torch-FL/blob/main/docs/vendors/ppu/installation.md)。这些资料说明兼容机制，不代替本机器算子和多卡验证。

## ClearML命令

更新远程代码仓库后，工作目录设为仓库根目录，单节点分配两张PPU。使用用户报告中的原镜像解释器：

```bash
RFT_PYTHON=/usr/local/bin/python bash scripts/navsim/debug_2ppu.sh
```

项目现在随仓库携带固定提交的 NAVSIM v1.1 源码，启动时会优先使用
`vendor/navsim`，因此不需要从 GitHub 下载 NAVSIM。源码提交由
`vendor/navsim/.vla_rft_commit` 校验。若使用外部源码目录，可在 ClearML 环境变量中设置
`RFT_NAVSIM_ROOT=/path/to/navsim-v1.1`；该目录必须是提交
`0811876c274e8b058ab2be9b3dcd4d37bd23f177`。

平台要求Python入口时填`scripts/navsim/debug_2ppu.py`，环境变量填`RFT_PYTHON=/usr/local/bin/python`，脚本参数留空。每个任务启动一次，不要在外层再套torchrun。基础镜像必须是用户已验证能识别PPU的镜像，不能换成通用CUDA容器。

数据目录仍是`/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1`，map为单数目录。输出在`/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs/smoke2-...`；下载cache在同级cache。

仅查看解析后的路径、预算和runtime，不安装、不训练：

```bash
RFT_PYTHON=/usr/local/bin/python bash scripts/navsim/debug_2ppu.sh --print-plan
```

## 安装与运行顺序

1. 检查基础镜像Python3.12、Torch2.6.0、TorchVision0.21.0，确认分配的PPU可见。
2. 在独立输出目录建立干净venv，不启用system-site-packages。通过符号链接复用原镜像torch/torchvision/torchaudio/triton包及其metadata，以及存在的nvidia命名运行库；保持SDK和动态库环境变量。不会修改链接指向的原文件。
3. 根据requirements-ppu.txt解析其他依赖，先保存pip dry-run计划，拒绝任何安装/替换Torch、Triton、CUDA运行库、FlashAttention、Xformers、PPU通信包等操作；按核查过的安装来源执行--no-deps安装，随后pip check。
4. 确认torch/torchvision实际加载的文件仍指向原镜像，导入真实Prismatic、FSQ和官方NAVSIM evaluator。
5. 两卡矩阵计算与NCCL all-reduce验证；随后实际执行小尺寸原FSQ、Llama、Flow/Sigma的合成输入反向、一次图像rollout、梯度同步、策略概率重算和checkpoint RNG恢复。合成探针不是驾驶训练成绩，也不包含预训练VLM或LPIPS。
6. 导出32训练/8验证场景，自动下载候选VLM和VGG/LPIPS权重，检查真实VLM图像输入。
7. 第一版tokenizer、WM、SFT各2步，图像RL和驾驶RL各1步，三组官方PDMS及质量诊断。每步保存。

骨干、动作编码、flow数学、奖励、第一版模型规模保持不变，未加入后续讨论的增强架构。少量步数只用于调试。

## 与旧依赖的差异

保留原镜像Torch2.6.0/TorchVision0.21.0，替代旧的普通Torch2.4.1/TorchVision0.19.1安装。其余包隔离安装，避免沿用镜像中diffusers/HF Hub、OpenCV/NumPy、rasterio/NumPy等已报告冲突。

Python3.12使用NumPy1.26.4、SciPy1.12.0、pandas2.2.3，替代旧Python3.10的NumPy1.23.4等版本。模型相关transformers4.40.1、diffusers0.30.3、timm0.9.10仍与第一版源码一致。nuPlan固定v1.2提交ce3c323af01c0d7ec5672f7832ef53f9c679aab0，不使用镜像里的nuPlan2.0。完整列表见requirements-ppu.txt。

固定的NAVSIM v1.1源码仍有np.int引用；仅在PPU隔离环境启动时补回其旧语义np.int=int。官方仓库及评分公式不修改。科学计算依赖变化仍需通过真实官方评测确认数值影响，不宣称与旧环境逐比特一致。

此venv链接到镜像中的厂商包，必须在相同镜像内使用，不能复制到没有PPU软件栈的机器。不要对该venv手动pip升级Torch等厂商包。

如果镜像预装了`nvidia-dali-cuda120`等可选DALI包，PPU隔离环境不会把它们桥接进venv。该包不是NAVSIM、nuPlan或VLA-RFT的依赖，且用户镜像中可能存在与`packaging`、`six`及DALI可选依赖不一致的元数据；Torch、TorchVision、Triton、NCCL和PPU包仍会严格保留并检查。

## 日志与恢复

- `launches/initial/node-0/ppu-environment/ppu-base.json`：原镜像版本和设备。
- 同目录`vendor-constraints.txt`、`pip-plan.json`、`ppu-imports.json`：版本保护、安装计划和实际导入记录。
- `launches/initial/nccl`：两卡通信检查。
- `launches/initial/ppu-kernels`：原模型合成探针、RNG恢复结果；其checkpoint不能当作真实训练初始化。
- `launches/initial/node-0/*.log`：各阶段日志。
- `tokenizer`、`wm`、`sft`、`rl_img`、`rl_drive`：真实数据训练checkpoint与逐rank指标。

恢复时保留同一镜像、相同代码/预算/卡数：

```bash
RFT_PYTHON=/usr/local/bin/python bash scripts/navsim/debug_2ppu.sh \
  --resume-run /mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs/原smoke2目录
```

全量16卡支持`train_16gpu.sh --runtime ppu`，拓扑参数沿用16卡文档；先完成两卡验证，不自动启动大作业。

## 验证边界

本地没有PPU和真实NAVSIM数据。安装计划、Python兼容性、源码导入/合成计算的本地验证记录会单独保存；服务器PPU算子、实际模型显存、真实VLM权重与官方PDMS仍由上述入口执行验证。任何阶段失败都会停止，不使用CPU或随机VLM冒充成功。
