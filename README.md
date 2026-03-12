# “未来杯”2025年第五届高校大数据挑战赛 A题：伪装目标检测

本项目是参加“未来杯”2025年第五届高校大数据挑战赛 A题（伪装目标检测）的参赛代码实现。

## 项目简介
本项目基于 **HitNet (High-resolution Iterative Feedback Network)** 架构实现，针对伪装目标检测（Camouflaged Object Detection, COD）中的多尺度识别、高分辨率推理以及模型泛化等核心问题进行了深度优化。

### 核心任务
1.  **基础检测 (Q1):** 在 CAMO 数据集上训练，实现对伪装物体的基本识别与定位。
2.  **多尺度检测 (Q2):** 针对 256x256 到 2048x2048 的不同分辨率输入，实现自适应检测策略。
3.  **泛化能力测试 (Q3):** 在 Camouflage-people 和 NC4K 数据集上测试跨域泛化性能，并输出检测框。

## 算法来源与复现
本项目复现并改进了 **HitNet** 模型，核心模型结构位于 `HitNet/` 目录下。

-   **原仓库地址:** [HUuxiaobin/HitNet](https://github.com/HUuxiaobin/HitNet)
-   **相关论文:** [High-resolution Iterative Feedback Network for Camouflaged Object Detection (AAAI 2023)](https://arxiv.org/abs/2203.11624)

## 目录说明
-   `HitNet/`: HitNet 模型核心实现逻辑。
-   `code/`: 比赛相关的训练、推理、数据增强及可视化脚本。
-   `docs/`: 包含赛题 PDF 及相关技术实施方案。
-   `README.md`: 本项目的主说明文档。

## 快速开始
具体的环境配置与运行指令，请参阅 [code/README.md](code/README.md)。

---
© 2025 “未来杯”高校大数据挑战赛
