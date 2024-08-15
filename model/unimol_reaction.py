import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from unimol_tools.data import DataHub
from unimol_tools.predictor import MolDataset
from unimol_tools.tasks.trainer import NNDataLoader


class UniMolR(nn.Module):
    def __init__(self, model, cls):
        super().__init__()
        self.model = model
        self.cls = cls

    def decorate_torch_batch(self, batch, device="cpu", task="repr"):
        """
        Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {
                k: v.to(device) for k, v in net_input.items()}, net_target.to(device)
        else:
            net_input, net_target = {'net_input': net_input.to(
                device)}, net_target.to(device)
        if task == 'repr':
            net_target = None
        elif task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def forward(self, batch_input, num_mols):
        datahub = DataHub(data=batch_input, task='repr', is_train=False, )
        dataset = MolDataset(datahub.data['unimol_input'])
        feature_name = None
        return_atomic_reprs = True
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=self.model.batch_collate_fn,
        )
        repr_dict = {"cls_repr": [], "atomic_coords": [], "atomic_reprs": [], "atomic_symbol": []}
        for batch in dataloader:
            net_input, _ = self.decorate_torch_batch(batch, device=next(self.model.parameters()).device)
            with torch.no_grad():
                outputs = self.model(**net_input,
                                return_repr=True,
                                return_atomic_reprs=return_atomic_reprs)
                assert isinstance(outputs, dict)
                repr_dict["cls_repr"].append(outputs["cls_repr"])
                if return_atomic_reprs:
                    repr_dict["atomic_symbol"].extend(outputs["atomic_symbol"])
                    repr_dict['atomic_coords'].extend(outputs['atomic_coords'])
                    # cls is important, may we cat to atomic_reprs
                    for i in range(len(outputs['atomic_reprs'])):
                        outputs['atomic_reprs'][i] = torch.cat([torch.unsqueeze(outputs["cls_repr"][i], dim=0), outputs['atomic_reprs'][i]])
                    repr_dict['atomic_reprs'].extend(outputs['atomic_reprs'])
        repr_dict["cls_repr"] = torch.concat(repr_dict["cls_repr"])

        if self.cls != 0:
            reaction_ = torch.split(repr_dict['cls_repr'], num_mols)
            batch_mask = torch.as_tensor(np.arange(max(num_mols)) < np.array(num_mols)[:, None], dtype=int, device=repr_dict['cls_repr'].device)
        else:
            new_num_mols = []
            n_ = 0
            for i in range(len(num_mols)):
                new_num_mols.append(0)
                for j in range(n_, n_+num_mols[i]):
                    new_num_mols[i] += len(repr_dict['atomic_reprs'][j])
                n_ += num_mols[i]
            atoms_reprs = torch.concat(repr_dict['atomic_reprs'])
            reaction_ = torch.split(atoms_reprs, new_num_mols)
            batch_mask = torch.as_tensor(np.arange(max(new_num_mols)) < np.array(new_num_mols)[:, None], dtype=int, device=atoms_reprs.device)

        """old, for inference np/list output, why uni-mol return this one??"""
        # unimol_repr = self.model.get_repr(batch_input, return_atomic_reprs=True)
        # if self.cls!=0:
        #     cls_repr = torch.as_tensor(unimol_repr['cls_repr'])
        #     reaction_ = torch.split(cls_repr, num_mols)
        #     batch_mask = torch.as_tensor(np.arange(max(num_mols)) < np.array(num_mols)[:, None], dtype=int, device=cls_repr.device)
        # else:
        #     new_num_mols = []
        #     n_ = 0
        #     for i in range(len(num_mols)):
        #         new_num_mols.append(0)
        #         for j in range(n_, n_+num_mols[i]):
        #             new_num_mols[i] += len(unimol_repr['atomic_reprs'][j])
        #         n_ += num_mols[i]
        #     atoms_reprs = torch.as_tensor(np.vstack(unimol_repr['atomic_reprs']))
        #     reaction_ = torch.split(atoms_reprs, new_num_mols)
        #     batch_mask = torch.as_tensor(np.arange(max(new_num_mols)) < np.array(new_num_mols)[:, None], dtype=int, device=atoms_reprs.device)

        cls_repr_batch = rnn_utils.pad_sequence(reaction_, batch_first=True, padding_value=0.)
        return cls_repr_batch, batch_mask


if __name__ == '__main__':
    from unimol_tools import UniMolRepr
    import random
    # single smiles unimol representation
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    smiles = 'c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]'
    smiles_list = [smiles]
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
    # CLS token repr
    print(np.array(unimol_repr['cls_repr']).shape)
    # atomic level repr, align with rdkit mol.GetAtoms()
    print(np.array(unimol_repr['atomic_reprs']).shape)

    training_data = {'atoms':[],
                    'coordinates':[],}
    num_mols = [5, 4, 8, 3]
    atoms = ["C", "H", "O"]
    for i in range(20):
        n_ = random.randint(4, 20)
        temp_a = []
        for j in range(n_):
            temp_a.append(random.choice(atoms))
        temp_c = np.random.randn(n_, 3)
        training_data["atoms"].append(temp_a)
        training_data["coordinates"].append(temp_c)

    unimol_repr = clf.get_repr(training_data, return_atomic_reprs=True)
    new_num_mols = []
    n_ = 0
    for i in range(len(num_mols)):
        new_num_mols.append(0)
        for j in range(n_, n_+num_mols[i]):
            new_num_mols[i] += len(unimol_repr['atomic_reprs'][j])
        n_ += num_mols[i]
    
    atoms_reprs = torch.as_tensor(np.vstack(unimol_repr['atomic_reprs']))
    reaction_atoms_ = torch.split(atoms_reprs, new_num_mols)

    cls_repr = torch.as_tensor(unimol_repr['cls_repr'])
    reaction_ = torch.split(cls_repr, num_mols)
    
    cls_repr_batch = rnn_utils.pad_sequence(reaction_, batch_first=True, padding_value=0.)
    batch_mask = torch.as_tensor(np.arange(max(num_mols)) < np.array(num_mols)[:, None], dtype=int)
    print()
