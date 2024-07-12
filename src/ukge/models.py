import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

class KGEModel(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        confidence_score_function: str = "logi",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels
        self.confidence_score_function = confidence_score_function
        self.node_emb = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)
        self.fc = torch.nn.Sequential(
            # torch.nn.Linear(1, 1, bias=True),
            # torch.nn.Linear(1, 32, bias=True),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Linear(32, 32, bias=False),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Linear(32, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        raise NotImplementedError
    
    def get_embeddings(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ):
        raise NotImplementedError
    
    def get_plausibility_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def get_confidence_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        raise NotImplementedError


class TransE(KGEModel):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        confidence_score_function: str = "logi",
        p_norm: float = 2.0,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse, confidence_score_function)
        self.p_norm = p_norm
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1, out=self.rel_emb.weight.data)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)
        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)
        return -((head + rel) - tail).norm(p=self.p_norm, dim=-1)
    
    def get_embeddings(
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
    
    def get_confidence_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor
    ) -> Tensor:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)
        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)
        plausibility_score = -((head + rel) - tail).norm(p=self.p_norm, dim=-1)
        if self.confidence_score_function == "cosine":
            return (F.cosine_similarity(head + rel, tail, dim=-1) + 1)/2
        elif self.confidence_score_function == "logi":
            return torch.sigmoid(self.fc(plausibility_score.view(-1, 1)).view(-1))
        elif self.confidence_score_function == "rect":
            return torch.clamp(self.fc(plausibility_score.view(-1, 1)).view(-1), min=0, max=1)
    
    def get_plausibility_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        return self.forward(head_index, rel_type, tail_index)


class DistMult(KGEModel):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        confidence_score_function: str = "logi",
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)
        self.confidence_score_function = confidence_score_function
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
        return (head * rel * tail).sum(dim=-1)

    def get_embeddings(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)
        return head, rel, tail
    
    def get_confidence_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor
    ) -> Tensor:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)
        plausibility_score = (head * rel * tail).sum(dim=-1)
        if self.confidence_score_function == "logi":
            return torch.sigmoid(self.fc(plausibility_score.view(-1, 1)).view(-1))
        elif self.confidence_score_function == "rect":
            return torch.clamp(self.fc(plausibility_score.view(-1, 1)).view(-1), min=0, max=1)

    def get_plausibility_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        return self.forward(head_index, rel_type, tail_index)
    
class ComplEx(KGEModel):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        confidence_score_function: str = "logi",
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse, confidence_score_function)
        self.node_emb_im = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb_im = Embedding(num_relations, hidden_channels, sparse=sparse)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.node_emb_im.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb_im.weight)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        rel_re = self.rel_emb(rel_type)
        rel_im = self.rel_emb_im(rel_type)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)
        return (self.triple_dot(head_re, rel_re, tail_re) +
                self.triple_dot(head_im, rel_re, tail_im) +
                self.triple_dot(head_re, rel_im, tail_im) -
                self.triple_dot(head_im, rel_im, tail_re))

    def get_embeddings(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        rel_re = self.rel_emb(rel_type)
        rel_im = self.rel_emb_im(rel_type)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)
        return head_re, head_im, rel_re, rel_im, tail_re, tail_im
    

    def get_confidence_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor
    ) -> Tensor:
        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        rel_re = self.rel_emb(rel_type)
        rel_im = self.rel_emb_im(rel_type)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)
        plausibility_score = (self.triple_dot(head_re, rel_re, tail_re) +
                              self.triple_dot(head_im, rel_re, tail_im) + 
                              self.triple_dot(head_re, rel_im, tail_im) -
                              self.triple_dot(head_im, rel_im, tail_re))
        if self.confidence_score_function == "logi":
            return torch.sigmoid(self.fc(plausibility_score.view(-1, 1)).view(-1))
        elif self.confidence_score_function == "rect":
            return torch.clamp(self.fc(plausibility_score.view(-1, 1)).view(-1), min=0, max=1)
    
    def triple_dot(self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
    ) -> Tensor:
        return (a * b * c).sum(dim=-1)
    

    def get_plausibility_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        return self.forward(head_index, rel_type, tail_index)


class RotatE(KGEModel):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        confidence_score_function: str = "logi",
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse, confidence_score_function)
        self.confidence_score_function = confidence_score_function
    #     self.node_emb_im = Embedding(num_nodes, hidden_channels, sparse=sparse) 
    # #     self.reset_parameters()

    # # def reset_parameters(self):
    # #     torch.nn.init.xavier_uniform_(self.node_emb.weight)
    # #     torch.nn.init.xavier_uniform_(self.node_emb_im.weight)
    # #     torch.nn.init.uniform_(self.rel_emb.weight, 0, 2 * math.pi)
    
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
        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        score = - torch.linalg.vector_norm(complex_score, dim=(1, 2))
        return score

    def get_embeddings(self, 
        head_index: Tensor, 
        rel_type: Tensor, 
        tail_index: Tensor
    ) -> Tensor:
        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)
        rel_theta = self.rel_emb(rel_type)
        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)
        return head_re, head_im, rel_re, rel_im, tail_re, tail_im


    def get_confidence_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor
    ) -> Tensor:
        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)
        rel_theta = self.rel_emb(rel_type)
        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)
        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        plausibility_score = - torch.linalg.vector_norm(complex_score, dim=(1, 2))
        if self.confidence_score_function == "logi":
            return torch.sigmoid(self.fc(plausibility_score.view(-1, 1)).view(-1))
        elif self.confidence_score_function == "rect":
            return torch.clamp(self.fc(plausibility_score.view(-1, 1)).view(-1), min=0, max=1)

    def get_plausibility_score(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        return self.forward(head_index, rel_type, tail_index)
        















if __name__ == "__main__":
    from ukge.datasets import KGTripleDataset
    from torch.utils.data import DataLoader
    
    train_data = KGTripleDataset(dataset='cn15k', split='train', num_neg_per_positive=10)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    
    pos_hrt, score, neg_hn_rt, neg_hr_tn = next(iter(train_dataloader))
    print(pos_hrt.shape, score.shape, neg_hn_rt.shape, neg_hr_tn.shape)
    
    print("=============================TransE=============================T")
    model = TransE(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128, confidence_score_function="rect")
    head, rel, tail = model.get_embeddings(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(head.shape, rel.shape, tail.shape)
    # plausibility_score = model(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    # print(plausibility_score)
    confidence_score = model.get_confidence_score(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(confidence_score)

    print("=============================DistMult=============================T")
    model = DistMult(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128, confidence_score_function="logi")
    head, rel, tail = model.get_embeddings(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(head.shape, rel.shape, tail.shape)
    # plausibility_score = model(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    # print(plausibility_score)
    confidence_score = model.get_confidence_score(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(confidence_score)

    print("=============================ComplEx=============================T")
    model = ComplEx(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128)
    head_re, head_im, rel_re, rel_im, tail_re, tail_im = model.get_embeddings(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(print(head_re.shape, head_im.shape, rel_re.shape, rel_im.shape, tail_re.shape, tail_im.shape))
    # plausibility_score = model(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    # print(plausibility_score)
    confidence_score = model.get_confidence_score(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(confidence_score)

    print("=============================RotatE=============================T")
    model = RotatE(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128)
    head_re, head_im, rel_re, rel_im, tail_re, tail_im = model.get_embeddings(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(print(head_re.shape, head_im.shape, rel_re.shape, rel_im.shape, tail_re.shape, tail_im.shape))
    # plausibility_score = model(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    # print(plausibility_score)
    confidence_score = model.get_confidence_score(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(confidence_score)