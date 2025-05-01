import yaml
from dataclasses import dataclass

@dataclass
class Config:
    # general
    train_batch_size: int
    val_batch_size: int
    num_epochs: int
    learning_rate: float
    output_dir: str

    # audio / text fine-tuning
    model_name_audio: str
    model_name_text: str

    # contrastive head
    projection_hidden_dim: int
    projection_dim: int
    projection_dropout: float

    # classification head
    phq_csv_path: str
    sampling_rate: int

def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)