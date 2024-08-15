from pretraining import *
from model.utils_ds import *
import pandas as pd
from lightning.pytorch import seed_everything
import json
import ipdb
from tqdm import tqdm


class Finetuner:
    def __init__(self, args):
        self.args = args
        self.data_prefix = "/amax/data/shirunhan/reaction/data"
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        self.dirpath = os.path.join("/amax/data/group_0/models/finetuning", 
                                    f"{args.ds}/{args.ft_type}_{args.batch_size}_{args.max_epochs}_{args.dropout}_{args.init_lr}_{args.min_lr}_{args.weight_decay}_{args.gtm}_{args.lm}")
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        self.logger = create_file_logger(os.path.join(self.dirpath, "log.txt"))

        s2p = "/amax/data/group_0/yield_data/pretraining_data/smiles2pos_path.pkl"
        s2p_ds = "/amax/data/group_0/yield_data/ds/smiles2pos_path.pkl"
        with open(s2p, "rb") as f:
            self.pos_dict = pickle.load(f)
        with open(s2p_ds, "rb") as f:
            self.pos_dict.update(pickle.load(f))

    def train(self, training_data, test_data):
        # model = Blip2Stage1_ft.load_from_checkpoint(self.args.load_model_path, strict=False, args=self.args, map_location=f"cuda:{self.args.devices.split(',')[0]}")
        p_=os.path.join(self.dirpath, "epoch=128.ckpt")
        model = Blip2Stage1_ft.load_from_checkpoint(p_, strict=False, args=self.args, map_location=f"cuda:{self.args.devices.split(',')[0]}")
        callbacks = []
        callbacks.append(plc.ModelCheckpoint(dirpath=self.dirpath,
                                             monitor="val_loss",
                                             filename='{epoch:03d}', 
                                             every_n_epochs=1, 
                                             save_top_k=1,
                                             save_on_train_epoch_end=True))
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.dirpath)
        trainer = pl.Trainer(
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            precision=self.args.precision,
            max_epochs=self.args.max_epochs,
            accumulate_grad_batches=self.args.accumulate_grad_batches,
            check_val_every_n_epoch=self.args.check_val_every_n_epoch,
            callbacks=callbacks,
            logger=tb_logger,
            strategy="auto", #"ddp_find_unused_parameters_true", "auto"
            enable_checkpointing=True,
            )

        # training_loader = DataLoader(training_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
        #                             shuffle=True, collate_fn=MyCollater(tokenizer=model.blip2qformer.tokenizer))
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                shuffle=False, collate_fn=MyCollater(tokenizer=model.blip2qformer.tokenizer))
        
        trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=test_loader)
        # model = Blip2Stage1_ft.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False, args=self.args)
        # p = trainer.predict(model=model, dataloaders=test_loader, ckpt_path="best")
        p = trainer.predict(model=model, dataloaders=test_loader)
        all_preds, all_labels = [], []
        for one in p:
            all_preds.append(one.preds)
            all_labels.append(one.labels)
        all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)
        print(max(all_preds), min(all_preds))
        res_ = {"rmse": torch.sqrt(self.mse(all_preds, all_labels)), 
                "mae": self.mae(all_preds, all_labels),
                "r2": self.r2(all_preds, all_labels)}
        return res_

    def print_results(self, all_best_p):
        res_ = {"rmse": [], "mae": [], "r2": []}
        res_float = {"rmse": [], "mae": [], "r2": []}
        for one in all_best_p:
            for k in one:
                res_[k].append(f"{float(one[k]*100):.4f}")
                res_float[k].append(float(one[k]*100))
        for k in res_:
            self.logger.info(res_[k])
            self.logger.info(f"{k}\t{np.mean(res_float[k]):.4f}\t{np.std(res_float[k]):.4f}")
    def get_BH_attention(self):
        data_list=[]
        out_list=[]
        cross_attention_list=[]
        p_=os.path.join(self.dirpath, "epoch=128.ckpt")
        model = Blip2Stage1_ft.load_from_checkpoint(p_, strict=False, args=self.args, map_location=f"cuda:{self.args.devices.split(',')[0]}")
        df_doyle = pd.read_excel(os.path.join(self.data_prefix, 'BH/Dreher_and_Doyle_input_data.xlsx'),
                                            sheet_name='FullCV_01', engine='openpyxl')
        raw_dataset = generate_buchwald_hartwig_rxns(df_doyle, 0.01)
        test_data = ReactionDataset(raw_dataset, pos_dict=self.pos_dict)
        # model = Blip2Stage1_ft.load_from_checkpoint(self.args.load_model_path, strict=False, args=self.args, map_location=f"cuda:{self.args.devices.split(',')[0]}")
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                shuffle=False, collate_fn=MyCollater(tokenizer=model.blip2qformer.tokenizer))
        model=model.cpu()
        for data in tqdm(test_loader):
            out,cross_attention=model.blip2qformer.get_attention(data)
            # ipdb.set_trace()
            data_list.append(data)
            out_list.append(out)
            cross_attention_list.append(cross_attention[10])
        return {"data":data_list,"out":out_list,"cross_attention":cross_attention_list}
            
    def run_BH_or_SM(self):
        """add split ratio"""
        if self.args.ds == "BH":
            num_ = int(3955*self.args.split[0])
            name_split = [('FullCV_{:02d}'.format(i), num_) for i in range(1, 11)]
        else:
            num_ = int(5760*self.args.split[0])
            name_split = [('random_split_{}'.format(i), num_) for i in range(10)]

        all_best_p = []
        # 10 splits
        for i, (name, split) in enumerate(name_split):
            ipdb.set_trace()
            # for parameter selection
            if i >= self.args.repeat:
                break
            if self.args.ds == "BH":
                df_doyle = pd.read_excel(os.path.join(self.data_prefix, 'BH/Dreher_and_Doyle_input_data.xlsx'),
                                            sheet_name=name, engine='openpyxl')
                raw_dataset = generate_buchwald_hartwig_rxns(df_doyle, 0.01)
            else:
                df = pd.read_csv(os.path.join(self.data_prefix, 'SM/{}.tsv'.format(name)), sep='\t')
                raw_dataset = generate_s_m_rxns(df, 0.01)
            training_data = ReactionDataset(raw_dataset[:split], pos_dict=self.pos_dict)
            test_data = ReactionDataset(raw_dataset[split:], pos_dict=self.pos_dict)

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(training_data, test_data)
            self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
            all_best_p.append(p)
        self.print_results(all_best_p)

    def run_BH_reactant(self):
        df_BH = pd.read_csv(os.path.join(self.data_prefix, "BH/BH.csv"), sep=',')
        dataset_BH = generate_buchwald_hartwig_rxns(df_BH, 0.01)
        with open(os.path.join(self.data_prefix, "BH/reactant_split_idxs.pickle"), "rb") as f:
            train_test_idxs = pickle.load(f)
        all_best_p = []
        # 5 random initialization
        for seed in range(min(self.args.repeat, 5)):
            seed_everything(seed=seed)
            training_data = []
            test_data = []
            for j in train_test_idxs[self.args.ds]["train_idx"]:
                training_data.append(dataset_BH[j])
            for j in train_test_idxs[self.args.ds]["test_idx"]:
                test_data.append(dataset_BH[j])
            
            self.logger.info(f"{seed}\tmae\trmse\tr2")
            p = self.train(ReactionDataset(training_data, pos_dict=self.pos_dict), ReactionDataset(test_data, pos_dict=self.pos_dict))
            self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
            all_best_p.append(p)
        self.print_results(all_best_p)

    def run_SM_xx(self):
        # test range
        name_split_dict = {"SM_test1": ('1', 4320, 5760), "SM_test2": ('2', 4320, 5760),
                           "SM_test3": ('3', 4320, 5760), "SM_test4": ('4', 4320, 5760)}

        (name, start, end) = name_split_dict[self.args.ds]
        self.args.name = name
        all_best_p = []
        # 5 random initialization
        for seed in range(self.args.repeat):
            seed_everything(seed=seed)
            df = pd.read_csv(os.path.join(self.data_prefix, 'SM/SM_Test_{}.tsv'.format(name)), sep='\t')
            raw_dataset = generate_s_m_rxns(df, 0.01)
            training_data = ReactionDataset(raw_dataset[:start], pos_dict=self.pos_dict)
            test_data = ReactionDataset(raw_dataset[start:], pos_dict=self.pos_dict)

            self.logger.info(f"{seed}\tmae\trmse\tr2")
            p = self.train(training_data, test_data)  
            self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
            all_best_p.append(p)
        self.print_results(all_best_p)

    def run_az(self):
        with open(os.path.join(self.data_prefix, "az/raw", "az_reactions_data.json"), "r") as f:
            raw_data_az_BH = json.load(f)
        dataset_az_BH = generate_az_bh_rxns(raw_data_az_BH)
        with open(os.path.join(self.data_prefix, "az/processed-0", "train_test_idxs.pickle"), "rb") as f:
            train_test_idxs = pickle.load(f)

        all_best_p = []
        # 10 splits
        for i in range(10):
            # for parameter selection
            if i >= self.args.repeat:
                break
            training_data = []
            test_data = []
            for j in train_test_idxs["train_idx"][i+1]:
                training_data.append(dataset_az_BH[j])
            for j in train_test_idxs["test_idx"][i+1]:
                test_data.append(dataset_az_BH[j])

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(ReactionDataset(training_data, pos_dict=self.pos_dict), ReactionDataset(test_data, pos_dict=self.pos_dict))
            self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
            all_best_p.append(p)
        self.print_results(all_best_p)


if __name__ == '__main__':
    args = get_args()
    
    args.ds = "BH"
    args.ft_type = "conformer"
    args.batch_size = 128
    args.max_epochs = 151
    args.dropout = 0.2
    args.weight_decay = 1e-4
    args.init_lr = 5e-4
    args.min_lr = 5e-5
    args.check_val_every_n_epoch = 1 
    args.warmup_steps = 0
    args.devices = "0,"
    args.repeat = 1
    args.cls = 0
    
    runner = Finetuner(args)
    save_dict=runner.get_BH_attention()
    with open("attention.pkl",'wb') as f:
        pickle.dump(save_dict,f)
    