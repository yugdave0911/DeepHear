from deephear.preprocessing import chunk_participant_data_multiple
from deephear.dataset import ContrastiveDataset
from deephear.models.models import ProjectionModule
from deephear.training.training import train_epoch, evaluate_epoch
from deephear.utils import get_device
from torch.optim import AdamW
from torch.utils.data import DataLoader

device = get_device()

audio_file_paths = "file_path_to_Audio"
transcript_file_paths = "file_path_to_transcript_data"

# 1. chunk your raw sessions
df_chunks = chunk_participant_data_multiple(
    audio_file_paths, transcript_file_paths,
    "audio_chunks/", "txt_chunks/"
)

# 2. build dataset + loader
ds = ContrastiveDataset(df_chunks, device=device)
dl = DataLoader(ds, batch_size=8, shuffle=True)

# 3. set up models + optimizer
audio_enc = ds.wav_proc.to(device)
text_enc  = ds.text_model.to(device)
audio_proj = ProjectionModule(768, 512).to(device)
text_proj  = ProjectionModule(768, 512).to(device)

def fine_tune_optimizer(mode: str, lr: float = 1e-4):
    """
    Returns an AdamW optimizer for either the audio or text encoder+projection heads.
    
    Args:
      mode: "audio" or "text"
      lr:   learning rate
    """
    if mode == "audio":
        params = list(audio_enc.parameters()) + list(audio_proj.parameters())
    elif mode == "text":
        params = list(text_enc.parameters()) + list(text_proj.parameters())
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'audio' or 'text'.")

    return AdamW(params, lr=lr)

# 4. train loop
for epoch in range(10):
    opt = fine_tune_optimizer("audio")
    tr_loss = train_epoch(audio_proj, text_proj, audio_enc, text_enc, dl, opt, device)
    val_loss = evaluate_epoch(audio_proj, text_proj, audio_enc, text_enc, dl, device)
    print(f"{epoch} train {tr_loss:.4f} val {val_loss:.4f}")