import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import argparse
from model.utils_ds import *
import logging


def create_file_logger(file_name: str = 'log.txt', log_format: str = '%(message)s', log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    handler = logging.FileHandler(file_name, "w")
    handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(log_level)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


class ReactionDataset(Dataset):
    def __init__(self, input_data, omit=True, pos_dict=None):
        super(ReactionDataset, self).__init__()
        self.omit = omit
        self.raw_data = input_data
        self.permutation = None
        self.mol_max_len = 10
        self.pos_dict = pos_dict
        self.can_dict = {}
    
    def shuffle(self):
        ## shuffle the dataset using a permutation matrix
        self.permutation = torch.randperm(len(self)).numpy()
        return self

    def __len__(self):
        return len(self.raw_data)

    def get_3d(self, index):
        confs = {"atoms": [], "coordinates": []}
        # a reaction may contain too many mols, limit it
        num_ = 0
        for s in self.raw_data[index][0].replace(">>", ".").split("."):
            m = Chem.MolFromSmiles(s)
            # omit single atom or not
            if self.omit:
                if m.GetNumAtoms() == 1:
                    continue
            num_ += 1
            if num_ > self.mol_max_len:
                break
            can_s = canonicalize_with_dict(s, self.can_dict)
            filename = self.pos_dict[s] if s in self.pos_dict else self.pos_dict[can_s]
            atom_temp = []                
            for atom in m.GetAtoms():
                atom_temp.append(atom.GetSymbol())
            confs["atoms"].append(atom_temp)
            confs["coordinates"].append(np.load(filename))
        return confs

    def get_1d(self, index):
        return self.raw_data[index][0]

    def __getitem__(self, index):
        ## consider the permutation
        if self.permutation is not None:
            index = self.permutation[index]
        return self.get_3d(index), self.get_1d(index), self.raw_data[index][1]


class MyCollater:
    def __init__(self, tokenizer, text_max_len=256):
        # self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def __call__(self, batch):
        conf_batch, text_batch, labels_y = zip(*batch)
        conf_input = {"atoms": [], "coordinates": []}
        num_mols = []
        for one in conf_batch:
            conf_input["atoms"].extend(one["atoms"])
            conf_input["coordinates"].extend(one["coordinates"])
            num_mols.append(len(one["atoms"]))

        text_tokens = self.tokenizer(text_batch,
                                     truncation=True,
                                     padding='longest',  # why originally use max_length in 3d-mol?
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True, 
                                     return_token_type_ids=False)
        
        labels_y = torch.as_tensor(labels_y)
        labels_y = labels_y.reshape(labels_y.shape[0], 1)
        return (conf_input, num_mols), text_tokens, labels_y
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', type=int, default=0)  # unimol, atom or cls

    # for ds dataset
    parser.add_argument('--ds', type=str, default="BH")
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--split", type=float, nargs='+', default=[0.7])

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gtm', action='store_false', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_false', help='use language modeling or not', default=True)
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--predict_hidden_size", default=256, type=int)
    parser.add_argument('--load_model_path', type=str, 
                        default="/checkpoint/True_True/pretraining_epoch=09-step=00570000.ckpt")
    parser.add_argument('--ft_type', type=str, default="merge", choices=["text", "conformer", "merge"])
    parser.add_argument('--data_prefix', type=str, default="/data/downstream")

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--use_3d', action='store_true', default=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1) 

    # GIN
    parser.add_argument('--gin_hidden_dim', type=int, default=300)
    parser.add_argument('--gin_num_layers', type=int, default=5)
    parser.add_argument('--drop_ratio', type=float, default=0.0)
    parser.add_argument('--tune_gnn', action='store_true', default=False)

    # train mode
    parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
    # evaluation
    parser.add_argument('--rerank_cand_num', type=int, default=64)
    # Bert
    parser.add_argument('--bert_hidden_dim', type=int, default=256, help='')
    parser.add_argument('--bert_name', type=str, default='yieldbert')
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--cross_attention_freq', type=int, default=2)
    parser.add_argument('--num_query_token', type=int, default=8)
    # optimization
    parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='optimizer min learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
    parser.add_argument('--warmup_steps', type=int, default=200, help='optimizer warmup steps')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
    parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
    parser.add_argument('--init_checkpoint', type=str, default='')
    parser.add_argument('--retrieval_eval_epoch', type=int, default=10)
    # data loader
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--match_batch_size', type=int, default=16)
    # parser.add_argument('--root', type=str, default='data/3d-mol-dataset/pubchem')
    parser.add_argument('--text_max_len', type=int, default=256)

    args = parser.parse_args()
    return args

