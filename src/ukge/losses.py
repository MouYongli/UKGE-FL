
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
    