import torch
from torch import Tensor
from ukge.models import KGEModel
from torch import Tensor

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
    neg_s_pred = model(neg_hn_rt[:,0])


    # Calculate the loss for positive head, positive tail, and negative head embeddings
    pos_h_loss = model.loss(pos_s_pred, pos_s)
    pos_t_loss = model.loss(pos_s_pred, pos_s)
    neg_h_loss = model.loss(neg_s_pred, neg_hr_tn)

    # Calculate the loss for PSL head, PSL tail, and negative tail embeddings
    psl_h_loss = model.loss(pos_s_pred, psl_s)
    psl_t_loss = model.loss(pos_s_pred, psl_s)
    neg_t_loss = model.loss(neg_s_pred, neg_hn_rt[:,1])

    # Combine the losses
    total_loss = pos_h_loss + pos_t_loss + neg_h_loss + psl_h_loss + psl_t_loss + neg_t_loss

    return total_loss

    
def compute_psl_loss(psl_prob, psl_score, prior_psl = 0.5, alpha_psl = 0.2):
    # prior_psl0 = torch.tensor(prior_psl)
    # psl_error_each = torch.square(torch.max(psl_score + prior_psl0 - psl_prob, torch.tensor(0)))
    # psl_mse = torch.mean(psl_error_each)
    # psl_loss = alpha_psl * psl_mse
    psl_loss = alpha_psl * torch.mean(torch.square(torch.max(psl_score + prior_psl - psl_prob, torch.tensor(0.0))))
    return psl_loss