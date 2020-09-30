import torch

def weightedSDR(output, target, x):
    noise = x - target
    expected_noise = x - output

    target_norm = torch.norm(target)
    target_norm_sq = target_norm * target_norm
    input_target_norm = torch.norm(x - target)
    input_target_norm_sq = input_target_norm * input_target_norm
    alpha = target_norm_sq / (target_norm_sq + input_target_norm_sq)

    loss = alpha * SDRLoss(output, target) + (1 - alpha) * SDRLoss(noise, expected_noise)

    return loss

def SDRLoss(output, target):
    output = output.view(-1, 16384)
    target = target.view(-1, 16384)

    dot_product = torch.sum(output * target)
    loss = (-1 * dot_product) / (torch.norm(target) * torch.norm(output))

    return loss