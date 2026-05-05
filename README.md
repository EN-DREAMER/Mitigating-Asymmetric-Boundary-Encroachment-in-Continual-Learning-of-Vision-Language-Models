# Mitigating Asymmetric Boundary Encroachment in Continual Learning of Vision-Language Models

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


## Running on Dataset

### Prepare the Dataset

To ensure a fair comparison and strictly follow the standard evaluation protocol, we adopt the dataset splits provided by previous work CoOp.
Please refer to their official guide for downloading and setting up the datasets: [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)

After setting up the datasets according to the guide above, please update the dataset paths in `xtail_ABE.py` and `xtail_ABE_16_shot.py` to match your local environment.

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
