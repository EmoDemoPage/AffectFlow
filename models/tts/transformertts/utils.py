import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).to(lengths.device)
    mask = (lengths.unsqueeze(1) <= ids).to(torch.bool)
    return mask


def reorder_batch(x, n_gpus):
    assert (
        x.size(0) % n_gpus
    ) == 0, "Batch size must be a multiple of the number of GPUs."
    new_x = x.new_zeros(x.size())
    chunk_size = x.size(0) // n_gpus

    for i in range(n_gpus):
        new_x[i::n_gpus] = x[i * chunk_size : (i + 1) * chunk_size]

    return new_x
