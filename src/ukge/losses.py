import torch
from torch import Tensor
from ukge.models import TransE, DistMult, ComplEx, RotatE
from torch.nn import functional as F


def compute_det_transe_loss(
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        neg_head_index: Tensor,
        neg_rel_type: Tensor,
        neg_tail_index: Tensor,
        model: TransE,
        margin: float = 1.0,
) -> Tensor:
    pos_score = model(head_index, rel_type, tail_index)
    neg_score = model(neg_head_index, neg_rel_type, neg_tail_index)
    return F.margin_ranking_loss(pos_score, neg_score, target=torch.ones_like(pos_score), margin=margin)

def compute_det_distmult_loss(
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        neg_head_index: Tensor,
        neg_rel_type: Tensor,
        neg_tail_index: Tensor,
        model: DistMult,
        margin: float = 1.0,
) -> Tensor:
    pos_score = model(head_index, rel_type, tail_index)
    neg_score = model(neg_head_index, neg_rel_type, neg_tail_index)
    return F.margin_ranking_loss(pos_score, neg_score, target=torch.ones_like(pos_score), margin=margin)
    
def compute_det_complex_loss(
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        neg_head_index: Tensor,
        neg_rel_type: Tensor,
        neg_tail_index: Tensor,
        model: ComplEx,
) -> Tensor:
    pos_score = model(head_index, rel_type, tail_index)
    neg_score = model(neg_head_index, neg_rel_type, neg_tail_index)
    scores = torch.cat([pos_score, neg_score], dim=0)
    pos_target = torch.ones_like(pos_score)
    neg_target = torch.zeros_like(neg_score)
    target = torch.cat([pos_target, neg_target], dim=0)
    return F.binary_cross_entropy_with_logits(scores, target)
    
def compute_det_rotete_loss(
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        neg_head_index: Tensor,
        neg_rel_type: Tensor,
        neg_tail_index: Tensor,
        model: RotatE,
        margin: float = 1.0,
) -> Tensor:
    pos_score = margin - model(head_index, rel_type, tail_index)
    neg_score = margin - model(neg_head_index, neg_rel_type, neg_tail_index)
    scores = torch.cat([pos_score, neg_score], dim=0)
    pos_target = torch.ones_like(pos_score)
    neg_target = torch.zeros_like(neg_score)
    target = torch.cat([pos_target, neg_target], dim=0)
    return F.binary_cross_entropy_with_logits(scores, target)

def compute_det_rotete_loss(
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        neg_head_index: Tensor,
        neg_rel_type: Tensor,
        neg_tail_index: Tensor,
        model: RotatE,
        margin: float = 1.0,
) -> Tensor:
    pos_score = margin - model(head_index, rel_type, tail_index)
    neg_score = margin - model(neg_head_index, neg_rel_type, neg_tail_index)
    scores = torch.cat([pos_score, neg_score], dim=0)
    pos_target = torch.ones_like(pos_score)
    neg_target = torch.zeros_like(neg_score)
    target = torch.cat([pos_target, neg_target], dim=0)
    return F.binary_cross_entropy_with_logits(scores, target)

def compute_psl_loss(psl_prob, psl_score, prior_psl = 0.5, alpha_psl = 0.2):
    psl_loss = alpha_psl * torch.mean(torch.square(torch.max(psl_score + prior_psl - psl_prob, torch.tensor(0.0))))
    return psl_loss

if __name__ == '__main__':
    from ukge.datasets import KGTripleDataset
    from torch.utils.data import DataLoader
    train_data = KGTripleDataset(dataset='cn15k', split='train', num_neg_per_positive=10)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    pos_hrt, score, neg_hn_rt, neg_hr_tn = next(iter(train_dataloader))    
    model = DistMult(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128)
    print(pos_hrt.shape, score.shape, neg_hn_rt.shape, neg_hr_tn.shape)
    head, rel, tail = model.get_embeddings(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    plausibility_score = model(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    print(head.shape, rel.shape, tail.shape)
    print(plausibility_score)