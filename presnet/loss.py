import torch


# Weighted BCE Loss for cost-sensitive classification
def weighted_bce_loss(logits, targets, weights):
    loss = (-weights * (torch.nn.functional.logsigmoid(logits) * targets +
                        torch.nn.functional.logsigmoid(-logits) * (1 - targets)))
    return loss.mean()


# L1 regularizer (can be added to loss to achieve sparsity in weights)
def l1_regularizer(model, lambda_l1=0.1):
    loss = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            loss += lambda_l1 * model_param_value.abs().sum()
    return loss
