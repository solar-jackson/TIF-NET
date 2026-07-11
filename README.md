# TIF-Net


## Project Structure

```angular2html
TIF-Net/
├── dataset/          # Dataset processing
├── networks/         # Network components
├── utils/            # Utility functions
├── manopth/          # MANO PyTorch implementation
├── model.py          # Main model
└── traineval.py      # Training & evaluation script
```

## Installation

```bash
pip install torch torchvision numpy scipy opencv-python
```

## Training

```bash
# HO3D dataset
python traineval.py --use_ho3d --network honet_transformer --epochs 70

# Dex-YCB dataset
python traineval.py --network honet_transformer --epochs 70
```

## Evaluation

```bash
python traineval.py --evaluate --resume ./exp-results/checkpoint.pth.tar
```

## Main Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_ho3d` | Use HO3D dataset | False |
| `--network` | Network architecture | honet_transformer |
| `--epochs` | Training epochs | 70 |
| `--lr` | Learning rate | 1e-4 |
| `--train_batch` | Training batch size | 64 |
| `--host_folder` | Results save directory | ./exp-results |
