# TIF-Net


## 项目结构

```angular2html
TIF-Net/
├── dataset/ # 数据集处理
├── networks/ # 网络组件
├── utils/ # 工具函数
├── manopth/ # MANO PyTorch 实现
├── model.py # 主模型
└── traineval.py # 训练评估脚本
```

## 安装

```bash
pip install torch torchvision numpy scipy opencv-python
```

## 训练

```bash
# HO3D数据集
python traineval.py --use_ho3d --network honet_transformer --epochs 70

# Dex-YCB数据集
python traineval.py --network honet_transformer --epochs 70
```

## 评估

```bash
python traineval.py --evaluate --resume ./exp-results/checkpoint.pth.tar
```

## 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_ho3d` | 使用HO3D数据集 | False |
| `--network` | 网络架构 | honet_transformer |
| `--epochs` | 训练轮数 | 70 |
| `--lr` | 学习率 | 1e-4 |
| `--train_batch` | 训练批次 | 64 |
| `--host_folder` | 结果保存目录 | ./exp-results |