import pickle
import pandas as pd
import os
from model.utils_ds import generate_buchwald_hartwig_rxns
import numpy as np
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit import Chem


if __name__ == '__main__':
    data_prefix = "/data/downstream"
    name = 'FullCV_01'
    df_doyle = pd.read_excel(os.path.join(data_prefix, 'BH/Dreher_and_Doyle_input_data.xlsx'),
                                sheet_name=name, engine='openpyxl')
    raw_dataset = generate_buchwald_hartwig_rxns(df_doyle, 0.01)
    s_list = set()
    for one in raw_dataset:
        for s in one[0].replace(">>", ".").split("."):
            s_list.add(s)

    s2p_ds = {}
    for id_, smi in enumerate(tqdm(s_list)):
        m = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m)
        m = Chem.RemoveHs(m)
        new_s = Chem.MolToSmiles(m)
        c = m.GetConformers()[0]
        p = c.GetPositions()
        p_ = os.path.join("/data/downstream/pos", f"{id_}.npy")
        np.save(p_, p)
        s2p_ds[new_s] = p_
    with open("/data/downstream/smiles2pos_path.pkl", "wb") as f:
        pickle.dump(s2p_ds, f)
