# 实验记录（2026-09-21）

本项目是**非官方迁移实现**。已实现 ≠ 已执行 ≠ 真实数据验证通过。以下没有官方NAVSIM复现成绩。

## 已执行

环境：macOS ARM CPU，Python3.9，Torch2.8.0；seed42；无CUDA。原仓库实际Prismatic HF model/processor成功导入，但缺预训练权重，未执行完整VLM forward。

| 检查 | 输入/规模 | 结果与证据 |
|---|---|---|
| 动作SE(2) | 转弯、π边界、训练归一化 | 通过，位置旋转与heading wrap、逆变换一致 |
| Eq8 / Eq12 / Eq13 | 显式数值测试 | 通过，去噪K平均、组减均值、clip-only目标；不是常规PPO替换 |
| 原DiT变体反向/RL | context16、hidden32、2层、3去噪步、2场景×3候选 | paper/source两种模式均通过有限梯度、参数更新、冻结encoder不变 |
| checkpoint精确恢复 | 完整状态+Adam+RNG | 恢复后采样链相同，下一optimizer更新后权重哈希相同 |
| 原FSQ tokenizer backward | 256px随机图、四层width32 | 实际源码forward/backward通过；非预训练tokenizer |
| WM MLE与连续rollout | 小Llama hidden32/1层；8物理未来帧 | 实际tokenize→MLE backward→optimizer→8帧自回归通过 |
| 合成奖励RL联通 | 2候选、L1-only合成smoke | 1次optimizer更新通过、冻结WM与tokenizer哈希不变；不是预训练LPIPS实验A |
| 编译、源码差异检查 | 新代码和脚本 | 语法检查、git diff --check已执行，见最终检查日志 |

日志：

- `outputs/local-checks/core-tests.log`：4项核心测试。
- `outputs/local-checks/world-smoke.log`：真实网络代码的合成测试。
- `outputs/local-checks/world-smoke/result.json`：机器可读记录。
- `outputs/local-checks/world-smoke/synthetic-comparison.gif`：合成真值、候选1、候选2。随机图片没有驾驶语义，不能用于判断模型质量。
- `outputs/local-checks/world-smoke/world-step-1.pt`、`policy-step-3.pt`：**合成测试checkpoint，不可作为真实RL初始化**（metadata不匹配，正式入口会拒绝）。
- `outputs/local-checks/world-smoke/tokenizer/`：随机FSQ测试权重，不能作为驾驶tokenizer。

合成smoke实际执行：tokenizer backward一次（未optimizer step）、WM optimizer 1步、策略SFT 2步、策略RL 1步；2候选、8帧、3去噪步；CPU约3.35秒，仅适用该小模型/合成数据。world loss=9.1361446；这不是图像质量或驾驶分数。详细结果以JSON为准。

## 已实现，未在真实数据或GPU验证

- 官方v1.1数据加载、日志级train/val隔离、未来图/轨迹对齐检查、训练统计。
- 预训练Prismatic接入和全VLM SFT、256px原FSQ+Llama世界模型训练。
- 完整A：冻结WM候选rollout + L1/预训练LPIPS + 组优势 + stochastic flow更新。
- 完整B：训练白名单场景官方PDMS奖励。
- 官方AbstractAgent / run_pdm_score接入、SFT/A/B同场景CSV比较。
- 真实图像单步/多步L1/LPIPS/PSNR、GIF；奖励/PDMS相关性与不一致案例输出。
- 8/16/32 GPU同步数据并行和多rank RNG保存；本地双进程Gloo梯度同步与恢复已通过，CUDA/NCCL、多机和性能仍未验证。
- Linux Python3.10依赖安装脚本；本地依赖测试不能代替该服务器环境验证。

## 尚未获得的实验结果

| 实验 | PDMS total/分项 | WM质量 | 有效真实场景 | GPU显存/耗时 | 实际真实训练steps |
|---|---|---|---|---|---|
| SFT baseline | 未运行 | 不适用 | 0 | 未测 | 0 |
| SFT + 图像RL A | 未运行 | 未测 | 0 | 未测 | 0 |
| SFT + PDMS RL B | 未运行 | 不适用 | 0 | 未测 | 0 |

图像奖励与PDMS相关性：未测。安全不一致案例：尚无真实数据案例，不编造。正式模型路径、正式评测结果路径：尚不存在。服务器按`minimal_server.sh`运行后将保存对应ckpt/CSV/JSON。

## 服务器验收顺序

1. 导出16训练+4验证，人工检查manifest token/相机timestamp证据及样例图；如只有frame association，不将其写成完整独立传感器同步证据。
2. preflight成功，确认完整VLM/tokenizer/VGG/LPIPS与NAVSIM固定commit。
3. WM与SFT真实样本反向（日志grad norm有限）；DiT零初始化使最初一步部分上游梯度为0，至少继续到后续步确认目标模块收到梯度。
4. WM连续rollout质量，保存单步与多步对比；短训练不证明模拟器有效。
5. SFT接入官方评测必须先成功，再运行A/B；检查候选多样性、同链概率误差、冻结模块哈希。
6. checkpoint恢复，再追加一步；对照相同配置和seed。
7. 扩展训练规模前分析WM动作越界率、奖励/PDMS不一致，并测量真实每步耗时和显存。不得用navtest调整阈值或选择候选。

本阶段没有完整训练时长承诺；按实测稳态step耗时计算预算。所有额外训练由用户显式运行，未自动消耗服务器GPU。

## 16卡启动入口的本地验证（2026-09-21）

新增 `train_16gpu.sh` / `train_16gpu.py`，通过共享CPFS协调单机16卡或双机各8卡。环境安装、权重下载、数据导出、tokenizer/WM/SFT/A/B训练与官方评测均有实际子进程调用。没有连接服务器执行这些阶段。

已执行：

- 两个CPU/Gloo进程：分rank梯度更新与单进程全批次一致；全局未使用参数保持不变；各rank随机数恢复；恢复后下一optimizer更新精确一致。证据 `local-validation/distributed-cpu-16gpu-launcher.json`。输入为合成小模型，不是真实NAVSIM或FSQ测试。
- Hydra 1.3.2：用12,000个token合成固定NAVSIM v1.1的官方缓存配置，token全部保留，历史4帧/未来10帧/Sequential worker。避免全量token作为命令行参数超长。这里只验证配置合成，没有生成官方cache。
- 4项既有动作/flow/RL/恢复核心测试通过。
- 小型随机原FSQ从checkpoint恢复HF目录，再加载后所有参数指纹相同。源码语法、三种启动计划、依赖版本一致性和完整checkpoint选择检查通过，见 `local-validation/launcher16-checks.json`。

未执行：Linux依赖完整安装、候选VLM真实权重forward、分布式FSQ+LPIPS反向、16 GPU NCCL训练、真实驾驶rollout、三组PDMS。真实数据步数仍为0，没有新增性能成绩。
