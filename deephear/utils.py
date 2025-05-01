import torch

def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def map_phq_score_to_label(phq_score: int) -> int:
    if 0 <= phq_score <= 4:
        return 0
    elif 5 <= phq_score <= 9:
        return 1
    elif 10 <= phq_score <= 14:
        return 2
    elif 15 <= phq_score <= 19:
        return 3
    else:
        return 4