# Mitigating Asymmetric Boundary Encroachment in Continual Learning of Vision-Language Models

> **NeurIPS 2026** | Mitigating catastrophic forgetting in vision-language continual learning by protecting historical decision boundaries against newly learned text features.

## Authors

**Mingfeng Li**<sup>1</sup>, **Xinyang Chen**<sup>1</sup>\*, **Xiucheng Li**<sup>1</sup>, **Weili Guan**<sup>2</sup>, **Liqiang Nie**<sup>1</sup>

<sup>1</sup> School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen)  
<sup>2</sup> School of Information Science and Technology, Harbin Institute of Technology (Shenzhen)

\* Corresponding author

## Links

- **Code Repository:** [GitHub](https://github.com/EN-DREAMER/NeurIPS26-ABE-CL-VLM)

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

> - [10/2026] Initial release

---

## Introduction

This repository provides the official implementation of **Mitigating Asymmetric Boundary Encroachment in Continual Learning of Vision-Language Models** (NeurIPS 2026).

In **Cross-domain Task-Agnostic Incremental Learning (X-TAIL)**, VLMs must acquire new concepts while preserving historical knowledge and zero-shot generalization. We identify **Asymmetric Boundary Encroachment (ABE)**, where newly optimized text features encroach upon historical class regions even when historical representations remain stable.

To address this issue, we propose **Analytic Decision Boundary (ADB)**, which enforces a statistics-derived geometric margin to protect historical decision boundaries against new text features. We further integrate Orthogonal Visual Adaptation (OVA) and Analytic Ridge Classifier (ARC) to enhance visual adaptability and joint inference, forming a unified framework for continual learning.

---


## Highlights

- **A new perspective on forgetting:** We identify ABE as a source of forgetting even when historical representations remain stable.
- **Analytic boundary protection:** ADB prevents new text features from encroaching on historical class regions.
- **Strong X-TAIL performance:** Our method achieves **74.4% / 84.4%** (16-shot) and **76.5% / 87.5%** (full-shot) in Average / Last accuracy.

---

## Method / Framework

Our framework centers on **Analytic Decision Boundary (ADB)**, which explicitly protects historical decision boundaries against **Asymmetric Boundary Encroachment (ABE)**.

Specifically, ADB models historical visual features using cumulative statistics and derives a **probabilistic margin** to constrain newly optimized text embeddings. To efficiently enforce this boundary, it employs **low-rank covariance approximation** and a hinge loss that penalizes potential encroachment into historical class regions.

Building upon ADB, we further integrate two complementary components:

- **Orthogonal Visual Adaptation (OVA):** Enables safe visual encoder adaptation through orthogonality-based regularization.
- **Analytic Ridge Classifier (ARC):** Enhances joint inference using multi-layer visual statistics and uncertainty-aware fusion.

These components share cumulative feature statistics, forming a unified framework that protects historical knowledge while improving adaptation to new concepts.

---

## Project Structure

```text
.
├── ABE_CL_VLM/
│   ├── clip/                         # CLIP model implementation
│   ├── scenario_datasets/            # Dataset loading and X-TAIL setup
│   ├── xtail_ABE.py                  # Full-shot training implementation
│   ├── xtail_ABE_16_shot.py          # 16-shot training implementation
│   ├── xtail_eval.py                 # Evaluation and joint inference
│   ├── run_seeds_16_shot_order_I.sh  # 16-shot experiments (Order I)
│   ├── run_seeds_16_shot_order_II.sh # 16-shot experiments (Order II)
│   ├── run_seeds_full_shot_order_I.sh  # Full-shot experiments (Order I)
│   ├── run_seeds_full_shot_order_II.sh # Full-shot experiments (Order II)
│   └── requirements.txt              # Python dependencies
├── README.md                          # Project documentation
└── LICENSE                            # License information
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/EN-DREAMER/ABE-CL-VLM.git
cd ABE-CL-VLM
```

### 2. Create the environment

```bash
conda create -n env python=3.10 -y
conda activate env
```

### 3. Install dependencies

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## Dataset / Benchmark

We evaluate on **X-TAIL**, which requires sequential learning across domains and classification over a joint label space of seen and unseen classes, without task identifiers at inference time.

The benchmark comprises ten datasets:

| Dataset | Dataset | Dataset | Dataset | Dataset |
| --- | --- | --- | --- | --- |
| Aircraft | Caltech101 | DTD | EuroSAT | Flowers |
| Food-101 | MNIST | OxfordPet | Stanford Cars | SUN397 |

We follow the class definitions and train/test splits used in **CoOp**. Please prepare the datasets according to the [CoOp dataset setup guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

### Task Orders

- **Order I:** Aircraft → Caltech101 → DTD → EuroSAT → Flowers → Food-101 → MNIST → Oxford Pets → Stanford Cars → SUN397.
- **Order II:** Stanford Cars → Aircraft → Oxford Pets → Food-101 → SUN397 → MNIST → Flowers → DTD → Caltech101 → EuroSAT.

Both task orders are evaluated under **16-shot** and **full-shot** settings.

### Evaluation Metrics

- **Transfer:** Zero-shot generalization to domains not yet learned.
- **Average:** Mean classification accuracy across domains and incremental steps.
- **Last:** Mean classification accuracy across domains after the final incremental step.

## Usage

### Reproducing the Experiments

From the repository root, run the corresponding script:

```bash
# 16-shot, Order I
./run_seeds_16_shot_order_I.sh

# 16-shot, Order II
./run_seeds_16_shot_order_II.sh

# Full-shot, Order I
./run_seeds_full_shot_order_I.sh

# Full-shot, Order II
./run_seeds_full_shot_order_II.sh
```

Each script runs the experimental pipeline, including training and evaluation, and outputs final accuracy metrics, as described in the original repository README.

## Experimental Results

### Main Results on X-TAIL (Order I)

The following scores (%) are reported in the paper. Our results are averaged over five random seeds.

| Setting | Method | Transfer ↑ | Average ↑ | Last ↑ |
| --- | --- | ---: | ---: | ---: |
| 16-shot | LADA | 61.5 | 72.4 | 83.0 |
| 16-shot | **Ours** | **63.4** | **74.4** | **84.4** |
| Full-shot | LADA | 61.9 | 75.1 | 86.9 |
| Full-shot | **Ours** | **63.1** | **76.5** | **87.5** |

The proposed framework improves on LADA across all three reported metrics in both settings. See **Tables 2–3** in the paper for comparisons with additional baselines and per-dataset results.


## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026mitigating,
  title={Mitigating Asymmetric Boundary Encroachment in Continual Learning of Vision-Language Models},
  author={Mingfeng Li and Xinyang Chen and Xiucheng Li and Weili Guan and Liqiang Nie},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgement

We thank the authors of [RAIL](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning) and [CoOp](https://github.com/KaiyangZhou/CoOp) for sharing their code, which served as references for our implementation.

## License

This project is released under the Apache License 2.0.
