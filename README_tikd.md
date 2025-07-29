# 📘 TIKD: Text-Image Joint Knowledge Distillation

This repository contains PyTorch code and pretrained models for:

**TIKD (Text-Image Joint Knowledge Distillation)** is a multi-modal knowledge distillation framework that enhances the training of vision transformers in resource-constrained scenarios by integrating textual semantics and visual features.

This method is proposed in the paper:

> **"TIKD: Where Text Meets Vision for Knowledge Distillation"**
> Xingzhu Liang, Chun Yin, Mengyuan Li, Yu-e Lin
> \[Anhui University of Science and Technology]
> \[[Paper Link](https://github.com/JSJ515-Group/TIKD)]

---

## 💡 Highlights

* 🚀 **Multi-modal Distillation**: Introduces semantically rich textual features into traditional visual knowledge distillation.
* 🧠 **Text-Teacher Module**: Uses MiniCPM-V 2.6 + CLIP to generate and encode image descriptions.
* 🔀 **Feature Fusion Module**: Dynamically fuses image and text features using cosine similarity-based weighting.
* 🎯 **Semantic-Guided Training**: Employs a staged training strategy to achieve cross-modal alignment.
* ✅ **Plug-and-Play**: Compatible with existing distillation strategies like KD, Hard KD, DKD, CTKD, PKCD.

---

## 🔍 Model Zoo

We provide student models distilled using TIKD:

| Model       | Dataset       | Acc\@1 | Acc\@5 | Params | Teacher Model | Distillation |
| ----------- | ------------- | ------ | ------ | ------ | ------------- | ------------ |
| TIKD-Large  | CIFAR-100     | 80.42  | 95.69  | 16.35M | ResNet32×4    | Hard KD      |
| TIKD-Medium | CIFAR-100     | 75.70  | 94.13  | 4.17M  | ResNet32×4    | Hard KD      |
| TIKD-Small  | CIFAR-100     | 65.20  | 89.77  | 1.09M  | ResNet32×4    | Hard KD      |
| TIKD        | Tiny ImageNet | 65.34  | 85.94  | 16.55M | ResNet32×4    | Hard KD      |

> *Note*: All models use MiniCPM-V for offline text generation and CLIP for text feature encoding.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/JSJ515-Group/TIKD.git
cd TIKD
pip install -r requirements.txt
```

Recommended environment:

* Python 3.8+
* PyTorch ≥ 1.10
* torchvision ≥ 0.11
* `transformers`, `timm`, `scikit-learn`, `matplotlib`

---

## 🗂️ Data Preparation

Prepare datasets such as CIFAR-10, CIFAR-100, or Tiny ImageNet. The directory structure should follow:

```
/path/to/data/
  train/
    class1/
    class2/
  val/
    class1/
    class2/
```

---

## ⚙️ Usage

### Inference with Pretrained Model

```bash
python main.py \
  --eval \
  --resume ./checkpoints/deit_tikd_base.pth \
  --data-path /path/to/cifar100 \
  --model deit_tikd_base
```

Expected output on CIFAR-100:

```
* Acc@1 80.42 Acc@5 95.69 Loss: 0.71
```

---

### 🔧 Training

Train with pre-generated text features and a visual teacher model:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
  --model tikd_base_patch_4_32 \
  --data-path /path/to/cifar100 \
  --batch-size 256 \
  --teacher-model resnet32x4 \
  --text-features ./text_features/ \
  --output_dir ./output/
```

> To use other distillation strategies (e.g., DKD, CTKD), set flags like `--distillation`, `--alpha`, and `--beta`.

---

### 🔍 Evaluation on Tiny ImageNet

```bash
python main.py \
  --eval \
  --resume ./checkpoints/deit_tikd_tinyimagenet.pth \
  --data-path /path/to/tinyimagenet \
  --model deit_tikd_base
```

---

## 📊 Visualization Tools

* 🔥 Grad-CAM attention heatmaps
* 🧩 t-SNE embeddings
* 📈 Training loss and accuracy curves

Run:

```bash
python visualize/attention_map.py
python visualize/tsne.py
```

## 🧪 Citation

If you use TIKD in your research, please cite:

```bibtex
@article{liang2025tikd,
  title={TIKD: Where Text Meets Vision for Knowledge Distillation},
  author={Liang, Xingzhu and Yin, Chun and Li, Mengyuan and Lin, Yu-e},
  journal={ArXiv preprint},
  year={2025},
  note={https://github.com/JSJ515-Group/TIKD}
}
```

## 🤝 Contributing

We welcome pull requests and issue submissions to improve the project.


## 📄 License

This project is released under the Apache 2.0 license.