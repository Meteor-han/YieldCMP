"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from utils import *
from model.pre_model_stage import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from lightning.pytorch import loggers as pl_loggers
# os.environ["CUDA_VISIBLE_DEVICES"]="1,"


if __name__ == '__main__':
    args = get_args()

    model = Blip2Stage1(args=args)
    dirpath = f"checkpoint/{args.gtm}_{args.lm}"
    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath=dirpath, 
                                         monitor="train_loss", 
                                         filename='pretraining_{epoch:02d}-{step:08d}', 
                                         every_n_train_steps=10000, 
                                         save_top_k=1, 
                                         save_on_train_epoch_end=True))
    # callbacks.append(LitProgressBar())
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=dirpath)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=tb_logger,
        strategy="auto", #"ddp_find_unused_parameters_true", "auto"
        enable_checkpointing=True,
        )
    
    # reactions
    with open("/amax/data/group_0/yield_data/pretraining_data/reactions.pkl", "rb") as f:
        data_ = pickle.load(f)
    data_y = []
    for one in data_:
        data_y.append((one, -1.))
    print(f"reactions nums: {len(data_y)}")

    # coordinates of molecules
    s2p = "/amax/data/group_0/yield_data/pretraining_data/smiles2pos_path.pkl"
    s2p_ds = "/amax/data/group_0/yield_data/ds/smiles2pos_path.pkl"
    with open(s2p, "rb") as f:
        pos_dict = pickle.load(f)
    with open(s2p_ds, "rb") as f:
        pos_dict.update(pickle.load(f))
    mydataset_ = ReactionDataset(data_y, pos_dict=pos_dict)
    
    training_loader = DataLoader(mydataset_, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=MyCollater(tokenizer=model.blip2qformer.tokenizer))
    # we use all reactions for training
    val_loader = DataLoader(ReactionDataset(data_y[:2000], pos_dict=pos_dict), batch_size=args.batch_size, 
                            shuffle=False, collate_fn=MyCollater(tokenizer=model.blip2qformer.tokenizer))
    
    trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=val_loader)

    print()
