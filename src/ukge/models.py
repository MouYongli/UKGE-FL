import math
from typing import Tuple

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

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ):
        r"""Returns the embedding of the head, relation and tail (for complEx, including re and im).

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError
    
    def score(self, 
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


class TransE(KGEModel):
    r"""The TransE model from the `"Translating Embeddings for Modeling
    Multi-Relational Data" <https://proceedings.neurips.cc/paper/2013/file/
    1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf>`_ paper.

    :class:`TransE` models relations as a translation from head to tail
    entities such that

    .. math::
        \mathbf{e}_h + \mathbf{e}_r \approx \mathbf{e}_t,

    resulting in the scoring function:

    .. math::
        d(h, r, t) = - {\| \mathbf{e}_h + \mathbf{e}_r - \mathbf{e}_t \|}_p

    .. note::

        For an example of using the :class:`TransE` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (int, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        p_norm (int, optional): The order embedding and distance normalization.
            (default: :obj:`1.0`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        p_norm: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.p_norm = p_norm
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight.data)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)
        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)
        return head, rel, tail
    
    def score(self, 
        head_index: Tensor, 
        rel_type: Tensor, 
        tail_index: Tensor
    ) -> Tensor:
        head, rel, tail = self.forward(head_index, rel_type, tail_index)
        return -((head + rel) - tail).norm(p=self.p_norm, dim=-1)


class RotatE(KGEModel):
    r"""The RotatE model from the `"RotatE: Knowledge Graph Embedding by
    Relational Rotation in Complex Space" <https://arxiv.org/abs/
    1902.10197>`_ paper.

    :class:`RotatE` models relations as a rotation in complex space
    from head to tail such that

    .. math::
        \mathbf{e}_t = \mathbf{e}_h \circ \mathbf{e}_r,

    resulting in the scoring function

    .. math::
        d(h, r, t) = - {\| \mathbf{e}_h \circ \mathbf{e}_r - \mathbf{e}_t \|}_p

    .. note::

        For an example of using the :class:`RotatE` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        margin: float = 1.0,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.node_emb_im = Embedding(num_nodes, hidden_channels, sparse=sparse) 
        self.margin = margin
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.node_emb_im.weight)
        torch.nn.init.uniform_(self.rel_emb.weight, 0, 2 * math.pi)

    
    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)
        rel_theta = self.rel_emb(rel_type)
        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)
        return head_re, head_im, rel_re, rel_im, tail_re, tail_im
    
    def score(self, 
        head_index: Tensor, 
        rel_type: Tensor, 
        tail_index: Tensor
    ) -> Tensor:
        
        head_re, head_im, rel_re, rel_im, tail_re, tail_im = self.forward(head_index, rel_type, tail_index)
        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        score = torch.linalg.vector_norm(complex_score, dim=(1, 2))
        return self.margin - score

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
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.reset_parameters()
        self.weights = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.model_type = model_type

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
   ) -> Tuple[Tensor, Tensor, Tensor]:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)
        return head, rel, tail
    
    def score(self, 
        head_index: Tensor, 
        rel_type: Tensor, 
        tail_index: Tensor
    ) -> Tensor:
        head, rel, tail = self.forward(head_index, rel_type, tail_index)
        # g = (head * rel).sum(dim=-1)
        # return torch.sigmoid(self.weights*g+self.bias) if self.model_type == "logi" else torch.clamp((self.weights*g+self.bias), min=0, max=1)
        return (head * rel * tail).sum(dim=-1)
        

if __name__ == "__main__":
    from ukge.datasets import KGTripleDataset
    from torch.utils.data import DataLoader
    
    train_data = KGTripleDataset(dataset='cn15k', split='train', num_neg_per_positive=10)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    pos_hrt, score, neg_hn_rt, neg_hr_tn = next(iter(train_dataloader))    
    model = DistMult(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128)
    print(pos_hrt.shape)
    head, rel, tail = model(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    plausibility_score = model.score(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(head.shape, rel.shape, tail.shape)
    print(plausibility_score)



