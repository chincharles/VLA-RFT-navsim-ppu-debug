# 论文—发布源码—迁移差异审计

依据：用户提供 `2510.00406v1.pdf` 第4–6页公式(3)–(13)；本地 `VLA-RFT-main` 快照。该快照没有 `.git`，不捏造其 upstream commit。独立仓库 `source-snapshot` 分支保存原始导入。源码链接以下均相对本仓库。

NAVSIM：[官方 v1.1 标签](https://github.com/autonomousvision/navsim/tree/0811876c274e8b058ab2be9b3dcd4d37bd23f177)，不是同名分支。所有源文件审计记录可由 `git diff source-snapshot` 查看。

| 模块 | 发布源码依据 | 论文／源码差异及迁移选择 | 分类 |
|---|---|---|---|
| VLM | `train/verl/vla-adapter/openvla-oft/prismatic/extern/hf/modeling_prismatic.py` | 复用原 vision backbone、multimodal projector、language model；初始前视图+官方 command vector 文本，24个新 query。去掉机器人 action-token label 掩码依赖，未来真值不进 encoder | NAVSIM 必要适配；query构造为重实现 |
| 原 VLM context | `verl/workers/actor/dp_actor.py:_forward_micro_batch` | 原提取视觉+动作隐藏状态；迁移 encoder 插入视觉 token 和24 learned queries后取相应隐藏状态，当前默认原分辨率由原 processor转换。没有保证与机器人输入协议逐token相同 | 适配 |
| 原 DiT | `prismatic/models/diffusion_transformer.py`、`transformer_utils.py` | 复制原 DiT/cross-attention，保留层结构，context_dim不再硬编码896，action input从7×H改3×H，输出3维，T=8。修正时间嵌入 batch广播、无效mask赋值。用等价本地Mlp替代timm导入 | 适配／修复 |
| FM监督(4) | `prismatic/models/action_heads.py:sample_noisy_actions` | 论文 `x=(1-t)eps+t*a`, target=`a-eps`；源码 target=`eps-a`。paper模式用真Beta(1.5,1)与正Euler步；source模式保留源码符号、uniform-power近似Beta及负步。原源码 `sample_beta` 不是严格Beta抽样 | 明示论文/代码分歧 |
| 时间轴 | 原 action_head 与 `verl/workers/rollout/hf_rollout.py` | 原 action_head sample_actions 从1向0，而实际RL rollout按k/K输入、负dt。source模式选择RL rollout路径，不声称修复它的训练时间语义。paper模式0→1。K=10是去噪步，与8×0.5s物理帧分开 | 明示分歧 |
| Sigma(6)(7) | `prismatic/models/noise_net.py:TokenSigmaNet` | 原输出是std和log_std，使用tanh映射log范围后exp，不是直接variance；协方差std²。迁移相同结构、范围0.08–0.2来自fsdp_workers初始化，fp32概率。没有额外sqrt(dt)乘子 | 保留机制／驾驶维度适配 |
| 概率(8)(9) | `verl/workers/actor/dp_actor.py` | 原代码对K求和，保留T×D逐维ratio；paper模式对T、D联合高斯求和，再对K取平均，得到一条候选一个ratio。T/D求和是对联合密度的明确解释。source模式保留K求和、T/D分量ratio | 明示分歧 |
| old/reference | `fsdp_workers.py:compute_log_prob/compute_ref_log_prob` | 本入口每批仅一次on-policy更新，采样后保存detached完整x_chain和old_logp，更新前重算误差检查。无需另复制old模型。默认无reference/KL（原启动脚本关闭）；reference不是old。没有实现可选reference-KL扩展 | 实现等价选择／未提供可选项 |
| 优势(12) | `verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage` | 源码除以组std（可选uniform_std）；paper仅减组均值。migration以显式B×N分组，禁止N<2；source采用组内无偏std。没有跨不同场景归一化 | 明示分歧 |
| 更新(13) | 同文件 `compute_policy_loss` | 源码dual-clip PPO(max/unclipped/clipped,负优势再dualclip=3)；paper直接clip(ratio)×Adv。两个独立分支，不静默套常规PPO | 明示分歧 |
| 辅助监督 | `dp_actor.py:update_policy` | 原可选MSE系数由ppo_kl门控，启动脚本max=.01，entropy=.003；paper模式固定可配置MSE+entropy。source模式额外保留0→.2的KL门控。该runner每批只更新一次，初始ratio=1时source gate常为0 | 记录实现和调度差异 |
| Dropout | 原 DiT attention / CrossAttention | 原有nn.Dropout和F.dropout。迁移全部关闭，使同一采样链新旧概率评估可重复；本地测试曾发现并修复函数式dropout残留 | 正确性修复，非新增探索奖励 |
| tokenizer | `ivideogpt/ctx_tokenizer/compressive_vq_model.py` | 保留原CompressiveFSQ tokenize/detokenize；256图像固定context1024/dynamic64。修正list FSQ levels加载时num_embeddings未定义。复用源码tokenizer，不宣称机器人checkpoint本身适用于驾驶 | 保留+修复 |
| WM动作条件 | `ivideogpt/processor.py:ContextMultiStepPredictionProcessor` | 保留独立context/dynamic/action vocab、256bin离散动作注入。新序列ctx+dyn(current)+action0+dyn(next)…，初始图像重复一次作为初始dynamic token；替换原机器人数据第一/第二帧隐含关系。训练和rollout共用`action_ids` | NAVSIM必要适配 |
| WM MLE(3) | processor labels；WM经HF causal model加载 | 新入口补齐条件交叉熵预训练，只对future visual tokens算loss；action token是给定条件，mask其loss。teacher forcing用真实历史token，rollout使用自己的token。loss按token均值，固定8帧64token时只是公式总和的固定缩放 | 预训练入口重实现 |
| WM生成 | 原vLLM rollout/`world_model_rollout`配置 | 新入口HF Llama KV cache，每次先采整个8步动作块，再自回归64图像token×8帧。默认greedy控制组内模拟噪声；源码启动脚本top_p=.8随机。不是每帧重规划，不是长时程闭环 | 实现/解码差异 |
| 参考图像(11) | `ray_trainer.py:msp_reward_fn`、`run_vla_rft.sh` | 发布启动脚本`use_img_gt_ac=True`、ctx_msp，比较专家动作生成rollout；论文公式正文写日志真值，前文允许两者。A默认expert_rollout；另配置log_future。两者绝不混称 | 保留发布参考配置／显式对照 |
| 奖励归约 | 同上 | 论文负L1+LPIPS时间sum；源码支持MSE/MAE、time mean/last/discount；yaml默认mse，但实际启动脚本显式设mae、mae=1、lpips=1、time mean。新A保留L1=1、LPIPS=1，按论文改time sum；source_math使用time mean，与发布启动脚本的奖励归约一致 | 明示分歧与用户要求的图像奖励实现 |
| LPIPS权重 | `ivideogpt/lpips.py:vgg16` | 原VGG路径占位且不存在时静默随机；迁移必须显式提供预训练VGG和LPIPS文件，校验linear MD5，无随机fallback | 严重正确性修复 |
| 坐标/归一化 | `NAVSIM/common/dataclasses.py:get_future_trajectory` | 原机器人动作不可复用。官方后轴SE2位姿→上一时刻局部SE2增量。heading wrap，inverse积分；训练集mean/std无clamp反归一化；仅WM bin超范围clamp并记录比例 | NAVSIM必要适配 |
| 正式输入与监督隔离 | `navsim_rft/data.py,agent.py` | inference只接收AgentInput，forward只输入image/text/state。未来图/专家轨迹在SFT、WM、reward、offline quality使用。测试缓存不进入RL白名单 | NAVSIM必要适配 |
| B驾驶奖励 | `navsim_rft/rewards.py:DrivingReward` | 用训练场景官方PDMS，sampling40×0.1s，默认官方scorer参数；不是论文方法。没有混合/安全门控，也没有图像熵奖励 | 用户要求的独立对照 |
| SFT和RL冻结 | `fsdp_workers.py` optimizer param_groups | 原RL optimizer仅head/projectors/sigma，不含VLM，尽管VLM forward存在。迁移明确冻结VLM与WM，反向不建其图，并检查参数hash；SFT默认更新VLM/query/flow/projector；Sigma冻结 | 保留原更新模块／强化检查 |
| 分布式/checkpoint | `verl/workers/fsdp_workers.py`、trainer | 原Ray、FSDP VLM、DDP头、vLLM WM和分片/adapter ckpt；新入口torchrun同步梯度数据并行，完整state+optimizer+RNG ckpt。没有移植Ray资源调度、vLLM、FSDP性能优化 | 可移植训练runner重实现；扩展规模未验证 |
| tokenizer从零训练 | 无完整发布训练入口 | 新`train_tokenizer.py`复用FSQ网络，L1+LPIPS重建，无GAN；架构width配置自行指定。不是原预训练recipe，也不宣称同等质量 | 缺失资源下的重实现 |
| 评测 | 官方`run_pdm_score.py` | 新AbstractAgent和运行时Hydra配置；一条固定seed ODE轨迹，不按测试reward选候选；拒绝遗漏/失败场景。v1 PDMS，不混v2 EPDMS | NAVSIM必要适配／明确推理协议 |

图像奖励只反映世界模型预测视觉结果与专家参考的接近程度。日志未来图像不是偏离专家候选的反事实真值；专家WM rollout也受同一模拟器偏差影响。二者均不证明碰撞安全。A使用PDMS独立评测、B单独使用训练PDMS优化，不能用B结果冒充原方法。

额外改进（混合奖励、安全门控、多视角融合、闭环重规划）均**未加入**。修复和工程实现差异不是新的研究效果主张。
