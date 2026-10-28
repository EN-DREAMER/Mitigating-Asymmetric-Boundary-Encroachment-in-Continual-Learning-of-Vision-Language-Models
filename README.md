# Mitigating Asymmetric Boundary Encroachment in Continual Learning of Vision-Language Models (NeurIPS 2026)


## Authors

Mingfeng Li¹, Xinyang Chen¹, Xiucheng Li¹, Weili Guan², and Liqiang Nie¹

¹ School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen)  
² School of Information Science and Technology, Harbin Institute of Technology (Shenzhen)

## Requirements

* Python 3.10
* PyTorch 2.4.1
* Torchvision 0.19.1
* Other dependencies are listed in requirements.txt.

To install requirements, run:

```sh
conda create -n env python=3.10 -y
conda activate env
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Datasets

Our X-TAIL experiments use ten datasets: FGVC Aircraft, Caltech101, DTD, EuroSAT, Oxford Flowers 102, Food-101, MNIST, Oxford-IIIT Pet, Stanford Cars, and SUN397. We follow the CoOp dataset setting, including its class definitions and train/test splits.

Please refer to the [CoOp dataset setup guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) for preparation instructions.

The paper evaluates two task orders:

* **Order I:** Aircraft → Caltech101 → DTD → EuroSAT → Flowers → Food-101 → MNIST → Oxford Pets → Stanford Cars → SUN397.
* **Order II:** Stanford Cars → Aircraft → Oxford Pets → Food-101 → SUN397 → MNIST → Flowers → DTD → Caltech101 → EuroSAT.

Both orders are evaluated in 16-shot and full-shot settings.

## Running

### Reproduction

To reproduce the main result in the paper, please run

```bash
cd ABE_CL_VLM

# run on 16-shot order-I setting
./run_seeds_16_shot_order_I.sh

# run on 16-shot order-II setting
./run_seeds_16_shot_order_II.sh

# run on full-shot order-I setting
./run_seeds_full_shot_order_I.sh

# run on full-shot order-II setting
./run_seeds_full_shot_order_II.sh
```

Each script will automatically run the full experimental pipeline, including training and evaluation, and output the final accuracy metrics.

## Acknowledgments

We thank the authors of [RAIL](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning) and [CoOp](https://github.com/KaiyangZhou/CoOp) for sharing their code, which served as references for our implementation.
