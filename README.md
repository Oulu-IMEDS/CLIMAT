# CLIMAT

CLIMAT: Clinically-Inspired Multi-Agent Transformers for Knee Osteoarthritis Trajectory Forecasting

Arxiv: https://arxiv.org/abs/2104.03642

## Installation
Go to the root of this repository
```bash
conda create -n climat python=3.7
conda activate climat 
```

## Training

Train CLIMAT models
```bash
python train.py config=seq_multi_prog_climat site=E n_meta_features=128
```

Train baselines on data from the sites A, B, C, and D (i.e., C is test site).
```bash
python train.py config=seq_multi_prog_fcn site=E n_meta_features=128
python train.py config=seq_multi_prog_gru site=E n_meta_features=128
python train.py config=seq_multi_prog_lstm site=E n_meta_features=128
python train.py config=seq_multi_prog_mmtf site=E n_meta_features=128
```

## Citation
Please cite the paper below if you find repo useful.
```
@inproceedings{nguyen2022climat,
  title={CLIMAT: Clinically-Inspired Multi-Agent Transformers for Knee Osteoarthritis Trajectory Forecasting},
  author={Nguyen, Huy Hoang and Saarakkala, Simo and Blaschko, Matthew B and Tiulpin, Aleksei},
  booktitle={2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
```
