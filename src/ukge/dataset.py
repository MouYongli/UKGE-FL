import os
import os.path as osp
from typing import Any, List, Tuple
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

here = osp.dirname(osp.abspath(__file__))
data_path = osp.join(here, '../data')

class KGDataset(Dataset):
    def __init__(self, root=data_path, 
                 dataset='cn15k', 
                 split='train',
                 transform=None, 
                 pre_transform=None):
        self.root=os.path.join(data_path, dataset)
        self.dataset = dataset
        self.split = split
        self.kg_triples_path = os.path.join(self.root, '{}.tsv'.format(split))
        self.entity_map_path = os.path.join(self.root, 'entity_id.csv')
        self.relation_map_path = os.path.join(self.root, 'relation_id.csv')

        self.entities = pd.read_csv(self.entity_map_path, header=None, names=["entity_name", "entity_id"])
        self.relations = pd.read_csv(self.relation_map_path, header=None, names=["entity_name", "entity_id"])
        self.triples = torch.tensor(pd.read_csv(self.kg_triples_path, sep='\t', header=None).to_numpy())
    
    def __getitem__(self, index) -> Any:
        return self.triples[index]
    
    def __len__(self):
        return len(self.triples)


class KGTripletLoader(DataLoader):
    def __init__(self, head_index: Tensor, 
                 rel_index: Tensor,
                 tail_index: Tensor,
                 scores: Tensor, 
                 **kwargs):
        self.head_index = head_index
        self.rel_index = rel_index
        self.tail_index = tail_index
        self.scores = scores
        super().__init__(range(head_index.numel()), collate_fn=self.sample, **kwargs)

    def sample(self, index: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        index = torch.tensor(index, device=self.head_index.device)
        head_index = self.head_index[index]
        rel_index = self.rel_index[index]
        tail_index = self.tail_index[index]
        score = self.scores[index]
        return head_index, rel_index, tail_index, score
    
if __name__ == "__main__":
    train_data = KGDataset()
    print(train_data.triples.shape)
    # train_loader = KGTripletLoader(head_index=train_data.edge_index[0],
    #                                rel_type=train_data.edge_type, 
    #                                tail_index=train_data.edge_index[1],
    #                                scores=train_data.scores[1],
    #                                batch_size=1000,
    #                                shuffle=True)
    # for head_index, rel_type, tail_index, scores in train_loader:
    #     print(head_index, rel_type, tail_index, scores)