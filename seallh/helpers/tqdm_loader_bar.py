import torch
from tqdm import tqdm

def tqdm_loader_bar(data_loader: torch.utils.data.DataLoader, 
                    mode: str, 
                    epoch: int,
                    max_epochs: int):

    if mode == 'validation':
        mode = 'val'

    assert epoch >= 0 or (epoch == -1 and mode == 'val'), "Epoch must be positive or -1 for validation without epoch info"
    assert max_epochs > 0, "Max epochs must be positive"

    if mode == 'train':
        desc = f'train'.ljust(10) + f'{epoch}/{max_epochs}'
    elif mode == 'val':
        split = f'val'.ljust(10)
        if epoch != -1:
            desc = split + f'{epoch}/{max_epochs}'
        else:
            desc = split
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    bar_format = '{l_bar}{bar}{r_bar}'
    loader_bar = tqdm(data_loader, 
                bar_format=bar_format, 
                total=len(data_loader),
                desc=desc)

    return loader_bar
