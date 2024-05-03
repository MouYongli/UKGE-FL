import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

class KGEModel(torch.nn.Module):
    r"""An abstract base class for implementing custom KGE models.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels

        self.node_emb = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)
        self.weights = torch.nn.Parameter(torch.zeros(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        r"""Returns the score for the given triplet.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError

class DistMult(KGEModel):
    r"""The DistMult model from the `"Embedding Entities and Relations for
    Learning and Inference in Knowledge Bases"
    <https://arxiv.org/abs/1412.6575>`_ paper.

    :class:`DistMult` models relations as diagonal matrices, which simplifies
    the bi-linear interaction between the head and tail entities to the score
    function:

    .. math::
        d(h, r, t) = < \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t >

    .. note::

        For an example of using the :class:`DistMult` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        model_type: str = "logi"
    ):
        
        assert model_type in ['logi', 'rect'], "Invalid value for 'model_types'. It should be one of 'logi', or 'rect'."

        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.model_type = model_type
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)
        f = (head * rel * tail).sum(dim=-1)
        return torch.sigmoid(self.weights*f+self.bias) if self.model_type == "logi" else torch.clamp((self.weights*f+self.bias), min=0, max=1)
    
    


if __name__ == "__main__":
    from ukge.datasets import KGTripleDataset
    from torch.utils.data import DataLoader
    train_data = KGTripleDataset(dataset='cn15k', split='train', num_neg_per_positive=10)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    pos_hrt, score, neg_hn_rt, neg_hr_tn = next(iter(train_dataloader))
    model = DistMult(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128)
    print(pos_hrt[:,0], pos_hrt[:,1], pos_hrt[:,2])
    pred_pos_score = model(pos_hrt[:,0], pos_hrt[:,1], pos_hrt[:,2])
    print(pred_pos_score.shape)
    pred_neg_hn__score = model(neg_hn_rt[:,:,0], neg_hn_rt[:,:,1], neg_hn_rt[:,:,2])
    print(pred_neg_hn__score.shape)
