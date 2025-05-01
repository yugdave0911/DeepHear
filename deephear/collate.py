import torch
from torch.nn.utils.rnn import pad_sequence

class Collate:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        feats, labels = zip(*batch)
        feats_p = pad_sequence(feats, batch_first=True)
        labels_t = torch.stack(labels)
        return feats_p.to(self.device), labels_t.to(self.device)