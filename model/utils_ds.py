from rdkit import Chem
from rdkit.Chem import rdChemReactions


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

    
def trans(smiles):
    # isomericSmiles, kekuleSmiles (F), canonical
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def canonicalize_with_dict(smi, can_smi_dict=None):
    if can_smi_dict is None:
        can_smi_dict = {}
    if smi not in can_smi_dict.keys():
        can_smi_dict[smi] = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    return can_smi_dict[smi]


# return: a list of tuples, [(SMILES_1, ..., SMILES_n, Yield), ...]
def generate_buchwald_hartwig_rxns(df, mul=0.01):
    """
    Converts the entries in the excel files from Sandfort et al. to reaction SMILES.
    """
    df = df.copy()
    fwd_template = '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>' \
                   '[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]'
    methylaniline = 'Cc1ccc(N)cc1'
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []
    for i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row['Aryl halide']), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])
    df['product'] = products
    rxns = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        can_smiles_dict[row['Aryl halide']] = aryl_halide
        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        can_smiles_dict[row['Ligand']] = ligand
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base
        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)
        can_smiles_dict[row['Additive']] = additive

        reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{additive}"
        rxns.append((f"{reactants}>>{row['product']}", row['Output'] * mul,))
    return rxns


def generate_s_m_rxns(df, mul=0.01):
    rxns = []
    for i, row in df.iterrows():
        rxns.append((row['rxn'].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in row['rxn'] else row['rxn'],) + (row['y'] * 100 * mul,))  # .replace(">>", ".").split(".")
    return rxns


def generate_az_bh_rxns_wo(raw_data, mul=0.01):
    rxns = []
    for one in raw_data:
        # all exists, no empty (""); raw data not canonicalized
        rxns.append((f'{one["reactants"][0]["smiles"]}.{one["reactants"][1]["smiles"]}.' + 
                     f'{one["reactants"][2]["smiles"]}.{one["base"]["smiles"]}.{one["solvent"][0]}>>' + 
                     f'{one["product"]["smiles"]}',) + (one["yield"]["yield"] * mul,))
    return rxns


def generate_az_bh_rxns(raw_data, mul=0.01):
    rxns = []
    for one in raw_data:
        # all exists, no empty (""); raw data not canonicalized
        rxns.append((f'{trans(one["reactants"][0]["smiles"])}.{trans(one["reactants"][1]["smiles"])}.' + 
                     f'{trans(one["reactants"][2]["smiles"])}.{trans(one["base"]["smiles"])}.{trans(one["solvent"][0])}>>' + 
                     f'{trans(one["product"]["smiles"])}',) + (one["yield"]["yield"] * mul,))
    return rxns


if __name__ == '__main__':
    import pickle
    import pandas as pd
    import os
    import json
    from collections import defaultdict

    """check 3d"""
    s2p = "/amax/data/shirunhan/reaction_mvp/data/smiles2pos_path.pkl"
    with open(s2p, "rb") as f:
        pos_dict = pickle.load(f)

    data_prefix = "/amax/data/shirunhan/reaction/data"
    name = 'FullCV_01'
    df_doyle = pd.read_excel(os.path.join(data_prefix, 'BH/Dreher_and_Doyle_input_data.xlsx'),
                                sheet_name=name, engine='openpyxl')
    raw_dataset = generate_buchwald_hartwig_rxns(df_doyle, 0.01)
    s_list = set()
    for one in raw_dataset:
        for s in one[0].replace(">>", ".").split("."):
            if s not in pos_dict:
                s_list.add(s)

    name = "random_split_0"
    df_0 = pd.read_csv(os.path.join(data_prefix, 'SM/{}.tsv'.format(name)), sep='\t')
    df_1 = pd.read_csv(os.path.join(data_prefix, 'SM/SM_Test_1.tsv'.format(name)), sep='\t')
    raw_dataset = generate_s_m_rxns(df_0, 0.01) + generate_s_m_rxns(df_1, 0.01)
    for one in raw_dataset:
        for s in one[0].replace(">>", ".").split("."):
            if s not in pos_dict:
                s_list.add(s)
    
    with open(os.path.join("/amax/data/shirunhan/reaction/data/az", "raw", "az_reactions_data.json"), "r") as f:
        raw_data_az_BH = json.load(f)
    raw_dataset = generate_az_bh_rxns(raw_data_az_BH)
    """check ELN BH, yields"""
    temp_ = defaultdict(list)
    multi_ = []
    for one in raw_dataset: 
        temp_[one[0]].append(one[1])
    for one in temp_: 
        if len(temp_[one]) > 1:
            multi_.append(temp_[one])
    """check ELN BH, OOS or not"""
    with open(os.path.join("/amax/data/shirunhan/reaction/data/az", "processed-0", "train_test_idxs.pickle"), "rb") as f:
        train_test_idxs = pickle.load(f)
    for i in range(10):
        training_data = []
        test_data = []
        for j in train_test_idxs["train_idx"][i+1]:
            training_data.append(raw_dataset[j])
        for j in train_test_idxs["test_idx"][i+1]:
            test_data.append(raw_dataset[j])
        training_mols = set()
        for one in training_data:
            for s in one[0].replace(">>", ".").split("."):
                training_mols.add(s)
        oos_count = 0
        for one in test_data:
            flag = False
            for s in one[0].replace(">>", ".").split("."):
                if s not in training_mols:
                    flag = True
            if flag:
                oos_count += 1
        print(oos_count, len(test_data))
    """
    118 225
    106 225
    117 225
    116 225
    126 225
    119 225
    134 225
    122 225
    117 225
    135 225
    """

    for one in raw_dataset:
        for s in one[0].replace(">>", ".").split("."):
            if s not in pos_dict:
                s_list.add(s)

    import numpy as np
    from rdkit.Chem import AllChem
    from tqdm import tqdm

    s2p_ds = {}
    for id_, smi in enumerate(tqdm(s_list)):
        m = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m)
        m = Chem.RemoveHs(m)
        new_s = Chem.MolToSmiles(m)
        c = m.GetConformers()[0]
        p = c.GetPositions()
        p_ = os.path.join("/amax/data/group_0/yield_data/ds/pos", f"{id_}.npy")
        np.save(p_, p)
        s2p_ds[new_s] = p_
    with open("/amax/data/group_0/yield_data/ds/smiles2pos_path.pkl", "wb") as f:
        pickle.dump(s2p_ds, f)
    print()
