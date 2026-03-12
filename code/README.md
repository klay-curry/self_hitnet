# COD 管道 (HitNet)

此文件夹包含围绕 HitNet 构建的 2025 年“未来杯”A 题（伪装目标检测）的比赛脚本。

## 快速开始
1) 安装依赖
```bash
pip install -r requirements.txt
```
2) 确保预训练的 PVT 权重文件存在于 `HitNet/pretrained_pvt/pvt_v2_b2.pth`。
3) 在 CAMO 数据集上训练（在code文件夹下运行）
```bash
python train.py --epochs 30 --batch-size 4 --train-size 352 --data-root code/assert/data/CAMO/CAMO-D --save-dir code/checkpoints
```
会根据年月日小时分钟输出到`code/checkpoints/*`下
4) 查找到三个问题的GT图像（在code文件夹下运行）
```bash
python match_ques_gt.py
```
5) 推理 / 可视化
```bash
# 问题一: CAMO 测试叠加 + 指标
python inference.py --task 1 --size 352 --checkpoint code/checkpoints/<模型名字.pth> --save-root code/answer
# 问题二: 多尺度 (256/512/1024/2048) 叠加
python inference.py --task 2 --size 352 --checkpoint code/checkpoints/<模型名字.pth> --save-root code/answer
# 问题三: 跨域伪装人群 + NC4K (掩码→边框)
python inference.py --task 3 --size 352 --checkpoint code/checkpoints/<模型名字.pth> --save-root code/answer
```
输出将存放在 `code/answer/<模型名字-时间戳>` 下。

对于`CAMO,NC4K,people`数据集的全量推理
```bash
# 全量推理 (含指标)
# CAMO
python inference.py --task 4 --size 352 --checkpoint code/checkpoints/<模型名字.pth> --save-root code/answer
# NC4K
python inference.py --task 5 --size 352 --checkpoint code/checkpoints/<模型名字.pth> --save-root code/answer
# 伪装人群
python inference.py --task 6 --size 352 --checkpoint code/checkpoints/<模型名字.pth> --save-root code/answer
```
## 注意事项
- 数据布局要求：
  - CAMO: `code/assert/data/CAMO/CAMO-D/train`、`test`，掩码位于 `code/assert/data/CAMO/gt`。
  - 伪装人群: 图像位于 `code/assert/data/Camouflage-people/CamouflageData/img`。
  - NC4K: 图像位于 `code/assert/data/NC4K/NC4K-D/{train,test}`，ID 的 JSON 文件。
- 训练使用加权 BCE+IoU（结构损失）跨 HitNet 阶段；验证报告 IoU/MAE。
- 推理在任务 3 中使用 TTA（翻转），在任务 2 中对大分辨率使用重叠切片。
