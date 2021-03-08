import numpy as np
import torch
import os
import pandas as pd
from collections import defaultdict
from rdkit import Chem

np.random.seed(2345) 


class SmilesEnumerator(object):
    '''
    Use the augmentation method presented at the following address:
    https://github.com/EBjerrum/SMILES-enumeration

    #Arguments
        charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
        pad: Length of the vectorization
        isomericSmiles: Generate SMILES containing information about stereogenic centers
        enum: Enumerate the SMILES during transform
        canonical: use canonical SMILES during transform (overrides enum)
    '''
    def __init__(self, charset = '@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.enumerate = enum
        self.isomericSmiles = isomericSmiles
        self.canonical = canonical


    @property
    def charset(self):
        return self._charset
        

    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        

    def fit(self, smiles, extra_chars=[], extra_pad = 5):
        '''Performs extraction of the charset and length of a SMILES datasets and sets self.pad and self.charset
        
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
            extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
            extra_pad: Extra padding to add before or after the SMILES vectorization
        '''
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad
        

    def randomize_smiles(self, smiles):
        '''Perform a randomization of a SMILES string
        must be RDKit sanitizable'''
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)


class CreateDatasets(object):
    ''' Use fingerprints and adjacency generation method presented at the following address:
    https://github.com/masashitsubaki/molecularGNN_smiles 
    '''
    def __init__(self, dataset, radius, device, use_augmentation):
        '''Initialize x_dict, in which each key is a symbol type
        (e.g., atom and chemical bond) and each value is its index.
        '''
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        self.use_augmentation = use_augmentation


    def create_dataset(self, filename, dataset, radius, device):
        dir_dataset = '../dataset/' + dataset + '/'
        '''Load a dataset.'''
        with open(dir_dataset + filename, 'r') as f:
            smiles_property = f.readline().strip().split()
            data_original = f.read().strip().split('\n')

        '''Exclude the data contains '.' in its smiles.'''
        data_original = [data for data in data_original
                            if '.' not in data.split()[0]]

        dataset = []
        for data in data_original:
            smiles, property = data.strip().split()

            '''Create each data with the above defined functions.'''
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = self.create_atoms(mol, self.atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = self.create_ijbonddict(mol, self.bond_dict)
            fingerprints = self.extract_fingerprints(radius, atoms, i_jbond_dict,
                                                self.fingerprint_dict, self.edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)

            '''Transform the above each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            '''
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
            property = torch.LongTensor([int(property)]).to(device)

            dataset.append((fingerprints, adjacency, molecular_size, property))

        return dataset

        
    def create_atoms(self, mol, atom_dict):
        '''Transform the atom types in a molecule (e.g., H, C, and O)
        into the indices (e.g., H=0, C=1, and O=2).
        Note that each atom index considers the aromaticity.
        '''
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [atom_dict[a] for a in atoms]
        return np.array(atoms)


    def create_ijbonddict(self, mol, bond_dict):
        '''Create a dictionary, in which each key is a node ID
        and each value is the tuples of its neighboring node
        and chemical bond (e.g., single and double) IDs.
        '''
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict


    def extract_fingerprints(self, radius, atoms, i_jbond_dict,
                            fingerprint_dict, edge_dict):
        '''Extract the fingerprints from a molecular graph
        based on Weisfeiler-Lehman algorithm.
        '''
        if (len(atoms) == 1) or (radius == 0):
            nodes = [fingerprint_dict[a] for a in atoms]
        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict
            for _ in range(radius):
                '''Update each node ID considering its neighboring nodes and edges.
                The updated node IDs are the fingerprint IDs.
                '''
                nodes_ = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    nodes_.append(fingerprint_dict[fingerprint])

                '''Also update each edge ID considering
                its two nodes on both sides.
                '''
                i_jedge_dict_ = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = edge_dict[(both_side, edge)]
                        i_jedge_dict_[i].append((j, edge))

                nodes = nodes_
                i_jedge_dict = i_jedge_dict_

        return np.array(nodes)


    def split_dataset(self, dataset, ratio):
        '''Shuffle and split a dataset.'''
        np.random.shuffle(dataset)
        n = int(ratio * len(dataset))
        return dataset[:n], dataset[n:]


    def get_Xy(self, cmpd_df, is_train):
        '''Extract smiles and labels of the group(train or test)'''
        df = cmpd_df[cmpd_df.group.eq('train' if is_train else 'test')]
        X = df.smiles
        y = df.activity.eq('active').astype(int).to_numpy()
        return X, y


    def save_Xy_data(self, csv_path, save_dir):
        '''Save as txt extension after given data preprocessing'''
        cmpd_df = pd.read_csv(csv_path)
        # df = pd.read_csv('cmpd_100.csv')

        X_train, y_train = self.get_Xy(cmpd_df, True)
        X_test, y_test = self.get_Xy(cmpd_df, False)

        sme = SmilesEnumerator()
        sme.fit(X_train)
        X_set, y_set = list(X_train), list(y_train)
        if self.use_augmentation:
            for (X, y) in zip(X_train, y_train): 
                for i in range(2):
                    X_set.append(sme.randomize_smiles(X))
                    y_set.append(y)

        data_train = {'smiles': np.array(X_set), 'activity': np.array(y_set)}
        data_train = pd.DataFrame(data=data_train)
        data_train = data_train.reindex(np.random.permutation(data_train.index))
        data_train.to_csv(f'{save_dir}/data_train.txt', index=False, header=None, sep=" ")
    
        data_test = {'smiles': np.array(X_test), 'activity': np.array(y_test)}
        data_test = pd.DataFrame(data=data_test)
        data_test.to_csv(f'{save_dir}/data_test.txt', index=False, header=None, sep=" ")


    def create_datasets(self, dataset, radius, device):
        ''''Create train, dev, test, N_fingerprints for training'''
        csv_path = '../dataset/bionsight/cmpd.csv'
        save_dir = '../dataset/bionsight'
        self.save_Xy_data(csv_path, save_dir)

        dataset_train = self.create_dataset('data_train.txt', dataset, radius, device)
        dataset_test = self.create_dataset('data_test.txt', dataset, radius, device)
        dataset_train, dataset_dev = self.split_dataset(dataset_train, 0.9)

        N_fingerprints = len(self.fingerprint_dict)

        return dataset_train, dataset_dev, dataset_test, N_fingerprints
