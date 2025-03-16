# MADGEN: Mass-Spec attends to De Novo Molecular generation

<a href="https://openreview.net/forum?id=770DetV8He"><img src="https://img.shields.io/badge/ICLR-2025-brown.svg" height=22.5></a>

Official implementation of [**MADGEN: Mass-Spec attends to De Novo Molecular generation**](https://openreview.net/forum?id=78tc3EiUrN&noteId=6vcJK7gC1B) by Yinkai Wang, Xiaohui Chen, Liping Liu, and Soha Hassoun.

<img src="assets/madgen.png">

## Environment Setup

```shell
conda create --name madgen python=3.9 rdkit=2023.09.5 -c conda-forge -y
conda activate madgen
pip install -r requirements.txt
```
## Data
The processed data is available in the zenodo repository: [MADGEN](https://zenodo.org/records/15036069).

```shell
wget https://zenodo.org/records/15036069/files/msgym.pkl?download=1 -O ./data/msgym/raw/msgym.pkl

wget https://zenodo.org/records/15036069/files/canopus.pkl?download=1 -O ./data/canopus/raw/canopus.pkl
```

The NIST is a commercial dataset.

## Training

```shell
python train.py --config configs/{dataset_name}.yaml --model Madgen
```

## Sampling

```shell
CUDA_VISIBLE_DEVICES=1 python sample.py \
       --config configs/{dataset_name}.yaml \
       --checkpoint {checkpoint_path}\
       --samples samples \
       --model Madgen \
       --mode test \
       --n_samples 50 \
       --n_steps 100 \
       --table_name {table_name} \
       --sampling_seed 42
```

## Evaluation

To run the evaluation, you now need to provide the file path as an argument:

```shell
python evaluation_generation.py --file_path /path/to/your/csvfile.csv
```

## Predictive Retrieval
For the predictive retrival, please refer to [JESTR](https://github.com/HassounLab/JESTR1).

## Contact

If you have any questions, please contact yinkai.wang@tufts.edu and soha.hassoun@tufts.edu.

## Citation

If you find this code useful for your research, please consider citing our paper:

```
@inproceedings{
       wang2025madgen,
       title={{MADGEN}: Mass-Spec attends to De Novo Molecular generation},
       author={Yinkai Wang and Xiaohui Chen and Liping Liu and Soha Hassoun},
       booktitle={The Thirteenth International Conference on Learning Representations},
       year={2025},
       url={https://openreview.net/forum?id=78tc3EiUrN}
}
```
