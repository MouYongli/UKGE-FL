import torch
from torch import Tensor
from ukge.models import KGEModel

def loss(
    model: KGEModel,
    pos_hrt: Tensor, 
    pos_s: Tensor, 
    neg_hn_rt: Tensor,
    neg_hr_tn: Tensor,
    psl_htr: Tensor,
    psl_s: Tensor,
    ) -> Tensor:
    pos_s_pred = model(pos_hrt[:,0], pos_hrt[:,1], pos_hrt[:,2])
    
def compute_psl_loss(psl_prob, psl_score, prior_psl = 0, _p_psl = 0.2):
    prior_psl0 = torch.tensor(prior_psl, dtype=torch.float32)
    psl_error_each = torch.square(torch.max(psl_score + prior_psl0 - psl_prob, torch.tensor(0)))
    psl_mse = torch.mean(psl_error_each)
    psl_loss = psl_mse * _p_psl
    return psl_loss