from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=True,
                      num_workers=num_workers)