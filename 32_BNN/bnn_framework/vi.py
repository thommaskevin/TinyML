# vi.py
import torch.nn.functional as F

def elbo_loss(output, target, model, kl_weight=1.0, likelihood='classification'):
    """
    Compute the Evidence Lower Bound (ELBO) loss.
    
    Args:
        output: model output
        target: ground truth
        model: BayesianModel instance (to access KL loss)
        kl_weight: scaling factor for KL term
        likelihood: one of 'classification', 'binary', 'regression'
    
    Returns:
        ELBO = negative log likelihood + kl_weight * KL
    """
    if likelihood == 'classification':
        nll = F.cross_entropy(output, target, reduction='sum')
    elif likelihood == 'binary':
        nll = F.binary_cross_entropy_with_logits(output.squeeze(), target.float(), reduction='sum')
    elif likelihood == 'regression':
        nll = F.mse_loss(output.squeeze(), target, reduction='sum')
    else:
        raise ValueError("likelihood must be 'classification', 'binary' or 'regression'")
    
    kl = model.kl_loss()
    return nll + kl_weight * kl