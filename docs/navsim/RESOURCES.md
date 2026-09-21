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
