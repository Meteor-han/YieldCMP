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
    print()
