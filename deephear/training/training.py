import torch
import torch.nn.functional as F

def contrastive_loss(a_emb, t_emb, temperature: float = 0.05):
    """
    InfoNCE Loss
    """
    a_norm = F.normalize(a_emb, p=2, dim=1)
    t_norm = F.normalize(t_emb, p=2, dim=1)
    sims = torch.matmul(a_norm, t_norm.T) / temperature
    labels = torch.arange(sims.size(0), device=sims.device)
    loss_a = F.cross_entropy(sims, labels)
    loss_t = F.cross_entropy(sims.T, labels)
    return (loss_a + loss_t) * 0.5

def train_epoch(audio_proj, text_proj,
                audio_encoder, text_encoder,
                dataloader, optimizer, device):
    audio_proj.train(); text_proj.train()
    audio_encoder.train(); text_encoder.train()
    total = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        # mean-pool across time dimension
        a_feat = audio_encoder(batch["audio"].to(device)).mean(1)
        t_feat = text_encoder(batch["text"].to(device)).mean(1)
        loss = contrastive_loss(audio_proj(a_feat), text_proj(t_feat))
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(dataloader)

def evaluate_epoch(audio_proj, text_proj,
                   audio_encoder, text_encoder,
                   dataloader, device):
    audio_proj.eval(); text_proj.eval()
    audio_encoder.eval(); text_encoder.eval()
    total = 0.0
    with torch.no_grad():
        for batch in dataloader:
            a_feat = audio_encoder(batch["audio"].to(device)).mean(1)
            t_feat = text_encoder(batch["text"].to(device)).mean(1)
            total += contrastive_loss(audio_proj(a_feat),
                                     text_proj(t_feat)).item()
    return total / len(dataloader)