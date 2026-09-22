# 资源核查

## 本地已找到

- `../VLA-RFT-main`：用户提供源码快照，没有Git metadata；没有适用的AGENTS.md。复制后创建独立源快照分支和迁移分支，不修改原目录。
- `../2510.00406v1.pdf`：已逐项读公式(3)、(4)、(8)、(11)、(12)、(13)。提取文本在工作区 `../output/audit/paper.txt`。
- 官方NAVSIM下载并固定v1.1标签提交 `0811876c274e8b058ab2be9b3dcd4d37bd23f177`。本地 `../NAVSIM-v1.1`。同名分支曾指向 `3e8291bfa89ff247231e0227778840cd0a036896`，已改为固定标签；不得混用。
- 本地独立测试环境 `../.venv-navsim`：macOS ARM CPU，Python3.9、Torch2.8.0、torchvision0.23.0、transformers4.40.1、diffusers0.30.3。不是服务器Python3.10/CUDA环境。

## 关键缺失，不能宣称端到端成功

| 资源 | 实际情况 | 对完成状态的影响 |
|---|---|---|
| NAVSIM logs / sensor_blobs / maps | 工作区未提供 | 未做真实场景时间戳、图像、轨迹读取；adapter代码已写，运行待服务器 |
| 训练、验证、测试 metric cache | 未提供 | 未执行PDMS训练奖励及官方评测 |
| VLA-RFT pretrained/RFT policy weights | checkpoints目录只有README，原README release TODO未完成 | 不能加载作者完整策略；需要用户提供兼容VLM，驾驶头重训 |
| 作者世界模型和tokenizer权重 | 工作区未提供，README列待发布 | 原tokenizer架构可用；提供FSQ重建训练重实现和Llama驾驶MLE入口；不能声称原权重复现 |
| 预训练VGG16和LPIPS | 未提供 | A完整图像奖励未实际运行。源码随机VGG fallback已禁止 |
| 原 tokenizer / policy SFT recipe | 发布入口以LIBERO RFT为主，没有可直接使用的NAVSIM或完整stage-I recipe | stage-I runner与tokenizer训练入口标重实现，网络尽量复用 |
| GPU / 服务器连接 | 本地无CUDA，用户将手动上传 | 未启动8/16/32卡任务，显存/耗时不能从CPU小模型外推为真实训练成绩 |

预训练权重的公开发布状态可能随时间改变；上表描述此次用户提供快照和实际本地资源，不断言作者永远不发布。原repo公开README查询时仍列政策/世界模型权重为待发布。

优先提供：兼容小型VLM目录、256px CompressiveFSQ tokenizer（或先训练）、NAVSIM navtrain/navtest数据与maps、预训练VGG/LPIPS。无需把服务器数据传回本机：按README在服务器预检和运行即可。

## 服务器启动补充（2026-09-21）

用户确认服务器已经下载全部 NAVSIM 数据，其他权重尚未下载。新增 `START_SERVER.md` 给出从只有数据开始的命令。数据和地图目录、GPU 型号仍未实际核实，服务器没有连接到本地会话。

- 从 LPIPS 作者仓库下载的 `lpips/weights/v0.1/vgg.pth` 已在本地验证 MD5=`d507d7349b931f0638a25a48a722f98a`。这只验证 linear 文件，VGG16 大权重和完整 LPIPS 前向仍未在本地执行。
- 已查到 [VLA-Adapter/LIBERO-Object](https://huggingface.co/VLA-Adapter/LIBERO-Object/tree/main) 的 HF 模型文件、processor/tokenizer 文件。模型卡声明 Prismatic/Qwen2.5-0.5B；它是候选初始化，绝非 VLA-RFT 作者 checkpoint。受网络连接限制，未获得其 config 内容及大模型权重，不能宣称已兼容。
- `scripts/navsim/check_vlm.py` 在服务器执行严格加载和真实训练样本前向，无 CUDA 或任何加载/前向失败时保存失败报告并非零退出。该脚本只完成了本地语法/帮助检查，未运行真实 VLM。
- `scripts/navsim/cache_subset.py` 向官方 cache 入口传实际 train/val token 合集，避免最小流程误启动全量 cache。官方 cache 计算仍待服务器运行。
