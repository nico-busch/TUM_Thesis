import torch


# Weighted BCE Loss for cost-sensitive classification
def weighted_bce_loss(logits, targets, weights):
    loss = (-weights * (torch.nn.functional.logsigmoid(logits) * targets +
                        torch.nn.functional.logsigmoid(-logits) * (1 - targets)))
    return loss.mean()


# Weighted hamming score (prescription accuracy)
def accuracy(logits, targets, weights):
    preds = logits.sigmoid() >= 0.5
    acc = weights[preds.eq(targets >= 0.5)].sum().true_divide(weights.sum())
    return acc


# Per class recall
def recall(logits, targets):
    preds = logits.sigmoid() >= 0.5
    rc = (preds.eq(targets >= 0.5) & (targets == 1)).sum(axis=0).true_divide((targets == 1).sum(axis=0))
    return rc


# Per class precision
def precision(logits, targets):
    preds = logits.sigmoid() >= 0.5
    pr = (preds.eq(targets >= 0.5) & (targets == 1)).sum(axis=0).true_divide((preds == 1).sum(axis=0))
    return pr


# L1 regularizer (can be added to loss to achieve sparsity in weights)
def l1_regularizer(model, lambda_l1=0.1):
    loss = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            loss += lambda_l1 * model_param_value.abs().sum()
    return loss
