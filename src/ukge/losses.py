import torch
from torch import Tensor
from torch.nn import functional as F


def compute_det_transe_distmult_loss(
        pos_score: Tensor,
        neg_score: Tensor,
        margin: float = 1.0,
        neg_per_positive: int = 10,
        verbose: bool = False,
) -> Tensor:
    if verbose:
        print(f"pos_score: {pos_score[:10]}", f"neg_score: {neg_score[:10]}")
    # 确保 neg_score 的大小是原始 pos_score 大小的 neg_per_positive 倍
    assert neg_score.size(0) == pos_score.size(0) * neg_per_positive, f"Expected neg_score size to be {pos_score.size(0) * neg_per_positive}, but got {neg_score.size(0)}"
    # 将 pos_score 重复指定次数
    pos_score = pos_score.repeat_interleave(neg_per_positive)
    # 创建与 pos_score 相同大小的 target
    target = torch.ones_like(pos_score)
    # 计算 margin ranking loss
    return F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)
    
def compute_det_complex_loss(
        pos_score: Tensor,
        neg_score: Tensor,
        verbose: bool = False,
) -> Tensor:
    if verbose:
        print(f"pos_score: {pos_score[:10]}", f"neg_score: {neg_score[:10]}")
    scores = torch.cat([pos_score, neg_score], dim=0)
    pos_target = torch.ones_like(pos_score)
    neg_target = torch.zeros_like(neg_score)
    target = torch.cat([pos_target, neg_target], dim=0)
    return F.binary_cross_entropy_with_logits(scores, target)

def compute_det_rotate_loss(
        pos_score: Tensor,
        neg_score: Tensor,
        margin: float = 1.0,
        verbose: bool = False,
) -> Tensor:
    pos_score = margin - pos_score 
    neg_score = margin - neg_score
    if verbose:
        print(f"pos_score: {pos_score[:10]}", f"neg_score: {neg_score[:10]}")
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
    from ukge.models import TransE, DistMult, ComplEx, RotatE
    train_data = KGTripleDataset(dataset='cn15k', split='train', num_neg_per_positive=10)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    
    transe = TransE(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128)
    distmult = DistMult(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128)
    complex = ComplEx(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128)
    rotate = RotatE(num_nodes=train_data.num_cons(), num_relations=train_data.num_rels(), hidden_channels=128)

    pos_hrt, score, neg_hn_rt, neg_hr_tn = next(iter(train_dataloader))
    neg_hn_rt = neg_hn_rt.view(-1, neg_hn_rt.size(-1))
    neg_hr_tn = neg_hr_tn.view(-1, neg_hr_tn.size(-1))

    print("=============================TransE=============================T")
    pos_score = transe(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    neg_score = transe(neg_hn_rt[:, 0], neg_hn_rt[:, 1], neg_hn_rt[:, 2])
    loss = compute_det_transe_distmult_loss(pos_score, neg_score)
    print(loss.item())

    print("=============================DistMult=============================T")
    pos_score = distmult(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    neg_score = distmult(neg_hn_rt[:, 0], neg_hn_rt[:, 1], neg_hn_rt[:, 2])
    loss = compute_det_transe_distmult_loss(pos_score, neg_score)
    print(loss.item())

    print("=============================ComplEx=============================T")
    pos_score = complex(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    neg_score = complex(neg_hn_rt[:, 0], neg_hn_rt[:, 1], neg_hn_rt[:, 2])
    loss = compute_det_transe_distmult_loss(pos_score, neg_score)
    print(loss.item())


    print("=============================RotatE=============================T")
    pos_score = rotate(pos_hrt[:, 0], pos_hrt[:, 1], pos_hrt[:, 2])
    neg_score = rotate(neg_hn_rt[:, 0], neg_hn_rt[:, 1], neg_hn_rt[:, 2])
    loss = compute_det_transe_distmult_loss(pos_score, neg_score)
    print(loss.item())

    