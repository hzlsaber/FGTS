<div align="center">
<h1> Rethinking Cross-Generator Image Forgery Detection through DINOv3 </h1>
</div>


[Zhenglin Huang](https://scholar.google.com/citations?user=30SRxRAAAAAJ&hl=en&oi=ao), [Jason Li](https://lxtgh.github.io/), [Haiquan Wen](https://orcid.org/0009-0009-3804-6753), [Tianxiao Li](https://tianxiao1201.github.io/), [Xi Yang](https://scholar.google.com/citations?user=ddfKpX0AAAAJ&hl=en), [Lu Qi](https://www.luqi.info/),
[Bei Peng](https://beipeng.github.io/), [Xiaowei Huang](https://cgi.csc.liv.ac.uk/~xiaowei/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/), [Guangliang Cheng](https://sites.google.com/view/guangliangcheng/homepage)

# FGTS: Fisher-Guided Token Selection for Deepfake Detection

A deepfake detection framework that leverages DINOv3 with intelligent token selection strategies for detecting AI-generated images.

## Overview

FGTS uses frozen DINOv3 to detect fake/synthetic images without large-scale fine-tuning. By analyzing individual transformer tokens and selecting the most discriminative ones using Fisher information scores, FGTS achieves **state-of-the-art cross-generator detection performance**.

### 🔥 Key Highlight

**Training with minimal data, detecting across diverse generators**: With just **1,000 real + 1,000 single-source fake images**, FGTS achieves SOTA performance in detecting images from **unseen generators** (Nano Banana, GPT-4o, etc.). This demonstrates exceptional generalization with minimal training overhead.

### Key Features

- **🚀 Minimal Training Data**: Achieves SOTA cross-generator detection with only 2K training samples (1K real + 1K single-source fake)
- **🎯 Superior Generalization**: Trained on one generator (e.g., ProGAN), generalizes to unseen generators ( Midjourney, Nano Banana, GPT-4o, etc.)
- **⚡ Training-Free Detection**: Classification using feature centroids (no training required)
- **🔧 Lightweight Linear Probe**: Optional supervised learning on frozen features with minimal parameters
- **🎲 Fisher-Guided Token Selection**: Automatically identifies the most discriminative tokens for detection
- **📊 Multiple Benchmarks**: Compatible with so-fake-ood, GenImage, and AIGCDetectionBenchmark datasets

## Installation

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)
- PyTorch >= 2.0.0

### Setup

```bash
# Clone the repository
git clone https://github.com/hzlsaber/FGTS.git
cd FGTS

# Create conda environment (recommended)
conda create -n FGTS python=3.10
conda activate FGTS

# Install PyTorch with CUDA support
# Visit https://pytorch.org/ to get the installation command for your CUDA version
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Download Data

We provide the training and validation sets used in this project for reproducibility:

- [**Training Set**](https://drive.google.com/file/d/1hnBWMVkZe0W60XGG_q8GYqnJGV8hTzQB/view?usp=drive_link)
- [**Validation Set**](https://drive.google.com/file/d/1cW_d7il4m01rwshlVIJFh0Cgu27BModF/view?usp=drive_link)
- [**So-Fake-OOD Test Set (Reference)**](https://drive.google.com/file/d/1WycDUXQSszOwFypvL33dXgFL8pQYzGA-/view?usp=drive_link)

For the remaining benchmarks, please download the test data from their **official project pages**:

- [**So-Fake-OOD**](https://github.com/hzlsaber/So-Fake)
- [**GenImage**](https://github.com/GenImage-Dataset/GenImage)
- [**AIGCDetectionBenchmark**](https://github.com/Ekko-zn/AIGCDetectBenchmark)

⚠ After downloading, please reorganize the data into the required directory structure shown below before running any scripts.

---

### Training/Validation Data

Following widely adopted conventions in image forensics (as seen in [CNNSpot](https://github.com/PeterWang512/CNNDetection/tree/master)), we place all fake training data under the `ProGAN` directory purely for a consistent project structure.

```
datasets/
├── train/progan/category (e.g., car)/
│   ├── 0_real/         # 1,000 real images (shared across all benchmarks)
│   ├── 1_fake_ldm/     # 1,000 LDM fakes (for so-fake-ood)
│   ├── 1_fake_sd14/    # 1,000 SD1.4 fakes (for GenImage)
│   └── 1_fake/         # 1,000 ProGAN fakes (for AIGCDetectionBenchmark)
└── val/progan/category (e.g., car)/
    ├── 0_real/         # Validation real images
    ├── 1_fake_ldm/     # Validation LDM fakes
    ├── 1_fake_sd14/    # Validation SD1.4 fakes
    └── 1_fake/         # Validation ProGAN fakes
```

### Benchmark-Specific Configurations

| Benchmark | Training Fake Source | Test Generators | Purpose |
|-----------|---------------------|-----------------|---------|
| [**so-fake-ood**](https://github.com/hzlsaber/So-Fake) | LDM (`1_fake_ldm`) | Ideogram, Nano_Banana, etc. | Test on latest commercial AIGC tools |
| [**GenImage**](https://github.com/GenImage-Dataset/GenImage) | SD1.4 (`1_fake_sd14`) | SD1.5, SD2.1, SDXL, etc. | Test modern diffusion model variants |
| [**AIGCDetectionBenchmark**](https://github.com/Ekko-zn/AIGCDetectBenchmark) | ProGAN only (`1_fake`) |  ProGAN, WFIR, SD1.4, etc. | Test GAN and diffusion model generalization |

**Key Insight**: Same real images (ProGAN/car), different fake sources allow fair comparison across benchmarks while matching test domain characteristics.

## Usage

### 1. Training-Free Detection

Perform deepfake detection **without any training** using feature centroids. Choose the appropriate `--reference_fake_type` based on your test benchmark:

**For so-fake-ood (train with LDM):**
```bash
python training_free_test.py \
    --model dinov3_vitl16 \
    --reference_dataset ./datasets/train/progan \
    --reference_category car \
    --reference_fake_type 1_fake_ldm \
    --test_base_dir ./datasets/test \
    --test_category car \
    --test_mode so-fake-ood \
    --token_strategy auto_fisher \
    --top_k 10 \
    --batch_size 32 \
    --output_dir ./results/training_free_sofakeood
```

**For GenImage (train with SD1.4), we show [GenImage_Tiny](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) here for reference:**
```bash
python training_free_test.py \
    --model dinov3_vitl16 \
    --reference_dataset ./datasets/train/progan \
    --reference_category car \
    --reference_fake_type 1_fake_sd14 \
    --test_base_dir /path/to/GenImage/1 \
    --test_mode GenImage \
    --token_strategy auto_fisher \
    --top_k 10 \
    --batch_size 32 \
    --output_dir ./results/training_free_genimage
```

**For AIGCDetectionBenchmark (train with ProGAN):**
```bash
python training_free_test.py \
    --model dinov3_vitl16 \
    --reference_dataset ./datasets/train/progan \
    --reference_category car \
    --reference_fake_type 1_fake \
    --test_base_dir /path/to/AIGCDetectionBenchmark/test \
    --test_mode AIGCDetectionBenchmark \
    --token_strategy auto_fisher \
    --top_k 10 \
    --batch_size 32 \
    --output_dir ./results/training_free_aigc
```

**Quick Start:**
```bash
cd examples
# Edit training_free_test.sh to set TEST_MODE
bash training_free_test.sh
```

### 2. Linear Probe Training

Train a lightweight linear classifier with **1K real + 1K fake images**. Use matching `--train_fake_type` and `--val_fake_type` for each benchmark:

**For so-fake-ood (train with LDM):**
```bash
python linear_probe_with_fisher.py \
    --model dinov3_vitl16 \
    --train_dataset ./datasets/train/progan \
    --val_dataset ./datasets/val/progan \
    --train_category car \
    --train_fake_type 1_fake_ldm \
    --val_fake_type 1_fake_ldm \
    --test_base_dir ./datasets/test \
    --test_category car \
    --test_mode so-fake-ood \
    --token_strategy auto_fisher \
    --top_k 10 \
    --max_train_samples 1000 \
    --max_test_samples 500 \
    --num_epochs 50 \
    --lr 0.01 \
    --batch_size 32 \
    --output_dir ./results/linear_probe_sofakeood
```

**For GenImage (train with SD1.4), we show [GenImage_Tiny](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) here for reference:**
```bash
python linear_probe_with_fisher.py \
    --model dinov3_vitl16 \
    --train_dataset ./datasets/train/progan \
    --val_dataset ./datasets/val/progan \
    --train_category car \
    --train_fake_type 1_fake_sd14 \
    --val_fake_type 1_fake_sd14 \
    --test_base_dir /path/to/GenImage/1 \
    --test_mode GenImage \
    --token_strategy auto_fisher \
    --top_k 10 \
    --max_train_samples 1000 \
    --max_test_samples 500 \
    --num_epochs 50 \
    --lr 0.01 \
    --batch_size 32 \
    --output_dir ./results/linear_probe_genimage
```

**For AIGCDetectionBenchmark (train with ProGAN only):**
```bash
python linear_probe_with_fisher.py \
    --model dinov3_vitl16 \
    --train_dataset ./datasets/train/progan \
    --val_dataset ./datasets/val/progan \
    --train_category car \
    --train_fake_type 1_fake \
    --val_fake_type 1_fake \
    --test_base_dir /path/to/AIGCDetectionBenchmark/test \
    --test_mode AIGCDetectionBenchmark \
    --token_strategy auto_fisher \
    --top_k 10 \
    --max_train_samples 1000 \
    --max_test_samples 6000 \
    --num_epochs 50 \
    --lr 0.01 \
    --batch_size 32 \
    --output_dir ./results/linear_probe_aigc
```

**Quick Start:**
```bash
cd examples
# Edit train_linear_probe.sh to set TEST_MODE
bash train_linear_probe.sh
```

### Key Parameters

| Parameter | Description | Options/Range |
|-----------|-------------|---------------|
| `--model` | DINOv3 model to use | `dinov3_vits16`, `dinov3_vitb16`, `dinov3_vitl16`, `dinov3_vith16`, `dinov3_vit_7b` |
| `--token_strategy` | Token selection method | `all`,  `patch`,  `auto_fisher`, `top_fisher`, `custom_indices` |
| `--top_k` | Number of tokens to select (for Fisher strategies) | Integer (e.g., 10, 20, 50) |
| `--test_mode` | Test dataset format | `so-fake-ood`, `GenImage`, `AIGCDetectionBenchmark` |
| `--batch_size` | Batch size for inference | Integer (default: 32) |
| `--img_size` | Input image resolution | Integer (default: 224) |

### Token Selection Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `all` | Use all tokens (CLS + registers + patches) | Baseline |
| `patch` | Use only patch tokens | Spatial information |
| `auto_fisher` | **Recommended**: Auto-compute Fisher scores and select top-k patch tokens | Best performance |
| `top_fisher` | Use pre-computed Fisher scores from file | Reuse previous analysis |
| `custom_indices` | Specify custom token indices | Manual token selection |

### Important Notes on Training Configurations

**Each benchmark requires its corresponding fake type:**

| Benchmark | `--reference_fake_type` / `--train_fake_type` | Training Data |
|-----------|----------------------------------------------|---------------|
| `so-fake-ood` | `1_fake_ldm` | 1K  real + 1K LDM fake |
| `GenImage` | `1_fake_sd14` | 1K  real + 1K SD1.4 fake |
| `AIGCDetectionBenchmark` | `1_fake` | 1K  real + 1K ProGAN fake |

**Why different fake types?**
- **so-fake-ood**: Uses LDM as a representative early diffusion model
- **GenImage**: Uses SD1.4 to match Stable Diffusion family
- **AIGCDetectionBenchmark**: Uses only ProGAN to test extreme cross-domain generalization

## Supported Models

### DINOv3 (via timm)

- `dinov3_vits16` - Small model (22M parameters)
- `dinov3_vitb16` - Base model (86M parameters)
- `dinov3_vitl16` - Large model (304M parameters)
- `dinov3_vith16` - Huge model (845M parameters)
- `dinov3_vit_7b` - Giant model (7B parameters)

## Results

### Performance on Benchmark Datasets

**Training Setup**: All linear probe models are trained with only **1,000 real + 1,000 single-source fake images**, then evaluated on **diverse unseen generators** (Nano Banana, Imagen4, Midjourney, Ideogram, etc.).

#### Training-Free Detection 

| Model | so-fake-ood (Acc/AUC) | GenImage (Acc/AUC) | AIGCDetectionBenchmark (Acc/AUC) |
|-------|----------------------|------------------------|----------------------------------|
| DINOv3-ViT-S16 |60.88/64.14 |43.19/40.93 |63.02/68.41 |
| DINOv3-ViT-B16 |60.71/65.02 |47.17/46.02 |66.48/76.87 |
| DINOv3-ViT-L16 |72.08/78.75 |61.83/68.61 |64.30/82.04 |
| DINOv3-ViT-H16 |73.36/82.57 |81.54/84.39 |67.97/84.03 |
| DINOv3-ViT-7B |**75.06/89.32** |**86.77/94.09** |**78.99/94.80** |

#### Linear Probe (Trained on 2K Images)

| Model | so-fake-ood (Acc/AUC) | GenImage (Acc/AUC) | AIGCDetectionBenchmark (Acc/AUC) |
|-------|----------------------|------------------------|----------------------------------|
| DINOv3-ViT-S16 |64.58/70.77 |48.56/46.47 |64.10/71.20 |
| DINOv3-ViT-B16 |70.31/77.95 |56.53/59.32 |70.38/81/32 |
| DINOv3-ViT-L16 |76.55/84.37 |70.13/78.64 |72.92/91.30 |
| DINOv3-ViT-H16 |77.81/88.03 |73.87/88.19 |80.45/94.82 |
| DINOv3-ViT-7B |**87.53/95.27** |**91.83/97.88** |**92.45/97.67** |

*Note: Metrics shown as Accuracy / AUC-ROC. All results demonstrate **strong cross-generator generalization** despite minimal training data.*

## Pretrained Weights

We provide pretrained linear probe weights trained on the three benchmark datasets. All models use **DINOv3-ViT-7B** backbone with Fisher-guided token selection.

### Available Checkpoints

The pretrained weights are included in this repository under the `checkpoints/` directory:

| Benchmark | Training Data | Checkpoint Path | Performance (Acc/AUC) |
|-----------|--------------|-----------------|----------------------|
| **so-fake-ood** | 1K real + 1K LDM fake | `checkpoints/so-fake-ood/linear_probe.pth` | 87.53 / 95.27 |
| **GenImage** | 1K real + 1K SD1.4 fake | `checkpoints/GenImage/linear_probe.pth` | 91.83 / 97.88 |
| **AIGCDetectionBenchmark** | 1K real + 1K ProGAN fake | `checkpoints/AIGCDetectionBenchmark/linear_probe.pth` | 92.45 / 97.67 |

After cloning the repository, you can directly use these checkpoints without any additional downloads.

### Using Pretrained Weights

To use the pretrained weights for inference, use the `--probe_checkpoint` argument:

**For so-fake-ood:**
```bash
python linear_probe_with_fisher.py \
    --model dinov3_vit_7b \
    --train_dataset ./datasets/train/progan \
    --probe_checkpoint ./checkpoints/so-fake-ood/linear_probe.pth \
    --test_base_dir ./datasets/test \
    --test_category car \
    --test_mode so-fake-ood \
    --batch_size 32 \
    --output_dir ./results/eval_sofakeood
```

**For GenImage:**
```bash
python linear_probe_with_fisher.py \
    --model dinov3_vit_7b \
    --train_dataset ./datasets/train/progan \
    --probe_checkpoint ./checkpoints/GenImage/linear_probe.pth \
    --test_base_dir /path/to/GenImage/1 \
    --test_mode GenImage \
    --batch_size 32 \
    --output_dir ./results/eval_genimage
```

**For AIGCDetectionBenchmark:**
```bash
python linear_probe_with_fisher.py \
    --model dinov3_vit_7b \
    --train_dataset ./datasets/train/progan \
    --probe_checkpoint ./checkpoints/AIGCDetectionBenchmark/linear_probe.pth \
    --test_base_dir /path/to/AIGCDetectionBenchmark/test \
    --test_mode AIGCDetectionBenchmark \
    --batch_size 32 \
    --output_dir ./results/eval_aigc
```

**Important Notes:**
- The checkpoint files contain both the linear probe weights and the Fisher-selected token indices
- No need to specify `--token_strategy` or `--top_k` when using `--probe_checkpoint` - these are loaded from the checkpoint
- Make sure to use the same model architecture (`dinov3_vit_7b`) as used during training
- Each checkpoint is specific to its benchmark due to different training fake types

## Project Structure

```
FGTS/
├── models/                      # Model implementations
│   └── dinov3_models.py        # DINOv3 wrapper
├── utils/                       # Utility functions
│   ├── data.py                 # Dataset loading
│   ├── features.py             # Feature extraction
│   ├── models.py               # Model loading interface
│   └── metrics.py              # Evaluation metrics
├── datasets/                    # Data directory
│   ├── train/
│   ├── val/
│   └── test/
├── examples/                    # Example scripts
│   ├── training_free_test.sh
│   └── train_linear_probe.sh
├── training_free_test.py       # Training-Free detection script
├── linear_probe_with_fisher.py # Linear probe training script
├── requirements.txt
└── README.md
```


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{your_paper,
  title={FGTS: Fisher-Guided Token Selection for Deepfake Detection},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [DINOv3](https://arxiv.org/abs/2304.07193) for self-supervised vision transformers
- [timm](https://github.com/huggingface/pytorch-image-models) for model implementations

## Contact

For questions or issues, please open an issue on GitHub or contact [zhenglin@liverpool.ac.uk](zhenglin@liverpool.ac.uk).
