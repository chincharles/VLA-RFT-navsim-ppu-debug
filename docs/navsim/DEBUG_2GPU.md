# 第一版：单机两卡调试

此入口保留第一版模型、数据编码和奖励目标，仅把自动流水线缩小到单机两张CUDA设备。没有加入后续讨论的时序融合、额外轨迹损失或新骨干。

此文件保留给普通CUDA环境。PPU-ZW810E请使用 [PPU_ZW810E.md](PPU_ZW810E.md) 中的 `debug_2ppu.sh`；PPU环境的 Torch 不能由旧安装脚本替换。

ClearML使用Linux、Python3.10，分配同一节点上的两张卡。工作目录为更新后的仓库根目录，每个任务只启动一次：

```bash
bash scripts/navsim/debug_2gpu.sh
```

平台要求Python入口时，填写`scripts/navsim/debug_2gpu.py`，参数留空。不要在外面再套torchrun。脚本不修改平台设置的CUDA_VISIBLE_DEVICES，使用其前两个逻辑设备。

数据路径已指定为`/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1`，读取trainval日志及传感器、map地图。自动安装环境、下载候选VLM与奖励权重、导出32个训练场景和8个验证场景，训练tokenizer/WM/SFT各2步、图像RL和驾驶RL各1步，执行三组官方PDMS及质量诊断。模型维度保持第一版，每卡1场景、每场景4个RL候选。短训练仅验证计算链路，不能反映收敛性能。

每步保存checkpoint。输出目录默认为`/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs/smoke2-<时间>-<随机标识>`，逐rank指标保存在各训练阶段目录，启动日志在`launches/initial/node-0`。下载缓存可复用之前任务的共享cache。

只查看配置，不安装或训练：

```bash
bash scripts/navsim/debug_2gpu.sh --print-plan
```

中断后保持相同代码和依赖，通过原输出目录恢复：

```bash
bash scripts/navsim/debug_2gpu.sh --resume-run /绝对路径/原smoke2目录
```

两卡任务不能直接作为16卡任务resume，因为当前checkpoint恢复要求world_size相同。16卡新训练仍使用`train_16gpu.sh`。

验证状态：本地仅验证入口参数、生成计划和分布式命令；此前双进程CPU/Gloo梯度与恢复测试通过。本次没有真实CUDA/NCCL、PPU或NAVSIM数据执行记录。候选VLM若加载不兼容会停止，不会自动换成随机网络。
