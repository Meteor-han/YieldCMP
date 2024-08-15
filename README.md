# YieldFCP

Data and codes for the paper "YieldFCP: Reaction Yield Prediction with Fine-grained 3D Cross-Modal Pre-training to Enhance Generalization Capability".

## Requirements

We implement our model on `Python 3.9.19`. These packages are mainly used:

```
rdkit                2024.3.3
torch                2.3.1
tensorboard          2.17.0
lightning            2.3.3
pytorch-lightning	 2.3.3
salesforce-lavis     1.0.2
unicore              0.0.1
unimol_tools         0.1.0.post1
rxnfp				 0.1.0
```

## Datasets

### Pre-training dataset

We utilize and filter reactions from USPTO and CJHIF. You can download USPTO from https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873 and CJHIF from https://github.com/jshmjs45/data_for_chem.

### Downstream dataset

We fine-tune our model on three publicly available downstream datasets. Related data and split for the HTE datasets (the Buchwald-Hartwig and the Suzuki-Miyaura reactions) and the real-world ELN dataset is stored in `downstream/BH`, `downstream/SM`, and `downstream/ELN`, respectively.

## Experiments

### Pre-training

Run `pretraining.py` to pre-train YieldFCP. For example,

```
python pretraining.py --max_epochs 10 --batch_size 8 --weight_decay 0.05 --init_lr 1e-4 --min_lr 5e-6

python pretraining.py --max_epochs 10 --batch_size 8 --cls 0 --lm --gtm --strategy_name ddp_find_unused_parameters_true
```

We provide the model with full combinations of CSC, CSM, and SG losses in `checkpoint`. 

### Fine-tuning

Run `finetuning.py` fine-tune YieldFCP on a given downstream dataset. For example,

```
python finetuning.py --devices 0, --batch_size 128 --ds BH --repeat 10 --max_epochs 150 --ft_type conformer --dropout 0.2 --weight_decay 1e-4 --init_lr 1e-4 --min_lr 1e-5 --check_val_every_n_epoch 1 --warmup_steps 0 --load_model_path checkpoint/True_True/pretraining_epoch=09-step=00570000.ckpt
```

