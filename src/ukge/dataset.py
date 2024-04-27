import os
import os.path as osp
from typing import Any, List, Tuple
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

here = osp.dirname(osp.abspath(__file__))
data_path = osp.join(here, '../..', 'data')

class KGTripleDataset:
    def __init__(self, root: str=data_path, dataset: str='cn15k', split: str='train', num_neg_per_positive: int=10):
        self.root=root
        self.dataset = dataset
        assert split in ['train', 'val', 'test'], "Invalid value for 'split'. It should be one of 'train', 'val', or 'test'."
        self.split = split
        self.num_neg_per_positive = num_neg_per_positive
        self.entity_id = pd.read_csv(os.path.join(self.root, dataset, 'entity_id.csv'))
        self.relation_id = pd.read_csv(os.path.join(self.root, dataset, 'relation_id.csv'))
        all_data_triples_df = pd.read_csv(os.path.join(self.root, dataset, 'data.tsv'), sep='\t', header=None)

        data_triples_df = pd.read_csv(os.path.join(self.root, dataset, '{}.tsv'.format(split)), sep='\t', header=None)
        self.data = {
            'head_index': data_triples_df[0].to_numpy(),
            'rel_index': data_triples_df[1].to_numpy(),
            'tail_index': data_triples_df[2].to_numpy(),
            'score': data_triples_df[3].to_numpy(),
        }

        # concept vocab
        self.cons = []
        # rel vocab
        self.rels = []
        # transitive rels vocab
        self.index_cons = {}  # {string: index}
        self.index_rels = {}  # {string: index}
        
        # Load data into cons and index_cons
        for _, row in self.entity_id.iterrows():
            # Add entity to cons list
            self.cons.append(row['entity string'])
            # Add entity and id mapping to index_cons dictionary
            self.index_cons[row['entity string']] = row['id']
        # Load data into rels and index_rels
        for _, row in self.relation_id.iterrows():
            # Add entity to cons list
            self.rels.append(row['relation string'])
            # Add entity and id mapping to index_cons dictionary
            self.index_rels[row['relation string']] = row['id']
        
        self.triples_record = set([])
        # head per tail and tail per head (for each relation). used for bernoulli negative sampling
        self.hpt = np.array([0])
        self.tph = np.array([0])
        tph_array = np.zeros((len(self.rels), len(self.cons)))
        hpt_array = np.zeros((len(self.rels), len(self.cons)))
        for h_, r_, t_, w in all_data_triples_df.to_numpy():  # only training data
            h, r, t = int(h_), int(r_), int(t_)
            tph_array[r][h] += 1.
            hpt_array[r][t] += 1.
            self.triples_record.add((h, r, t))
        self.tph = np.mean(tph_array, axis=1)
        self.hpt = np.mean(hpt_array, axis=1)

    def __len__(self):
        return len(self.data[self.split])

    def num_cons(self):
        '''Returns number of ontologies.

        This means all ontologies have index that 0 <= index < num_onto().
        '''
        return len(self.cons)

    def num_rels(self):
        '''Returns number of relations.

        This means all relations have index that 0 <= index < num_rels().
        Note that we consider *ALL* relations, e.g. $R_O$, $R_h$ and $R_{tr}$.
        '''
        return len(self.rels)

    def rel_str2index(self, rel_str):
        '''For relation `rel_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_rels.get(rel_str)

    def rel_index2str(self, rel_index):
        '''For relation `rel_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.rels[rel_index]

    def con_str2index(self, con_str):
        '''For ontology `con_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_cons.get(con_str)

    def con_index2str(self, con_index):
        '''For ontology `con_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.cons[con_index]
    
    def corrupt_pos(self, triple, pos):
        """
        :param triple: [h, r, t]
        :param pos: index position to replace (0 for h, 2 fot t)
        :return: [h', r, t] or [h, r, t']
        """
        hit = True
        res = None
        while hit:
            res = np.copy(triple)
            samp = np.random.randint(self.num_cons())
            while samp == triple[pos]:
                samp = np.random.randint(self.num_cons())
            res[pos] = samp
            if tuple(res) not in self.triples_record:
                hit = False
        return res
    
    # bernoulli negative sampling
    def corrupt(self, triple, neg_per_positive, tar=None):
        """
        :param triple: [h r t]
        :param tar: 't' or 'h'
        :return: np.array [[h,r,t1],[h,r,t2],...]
        """
        # print("array.shape:", res.shape)
        if tar == 't':
            position = 2
        elif tar == 'h':
            position = 0
        else:
            raise ValueError(f"value tar {tar} is not 't' or 'h'")
        res = [self.corrupt_pos(triple, position) for i in range(neg_per_positive)]
        return np.array(res)

    # bernoulli negative sampling on a batch
    def corrupt_batch(self, t_batch, neg_per_positive, tar=None):
        res = np.array([self.corrupt(triple, neg_per_positive, tar) for triple in t_batch])
        return res



class KGTripleLoader(DataLoader):
    def __init__(self, 
                 dataset: KGTripleDataset,
                 **kwargs):
        self.dataset = dataset
        super().__init__(range(self.dataset.len()), collate_fn=self.sample, **kwargs)


    def sample(self, index: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        index = torch.tensor(index, device=self.head_index.device)
        head_index = self.head_index[index]
        rel_index = self.rel_index[index]
        tail_index = self.tail_index[index]
        score = self.scores[index]
        return head_index, rel_index, tail_index, score
    
if __name__ == "__main__":
    train_data = KGTripleDataset(split='train')
    val_data = KGTripleDataset(split='val')
    test_data = KGTripleDataset(split='test')
    all_data = KGTripleDataset(split='all')
    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
    print(len(all_data))