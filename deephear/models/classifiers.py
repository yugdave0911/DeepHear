import torch.nn as nn
import torch.nn.functional as F

class ConvClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes=5, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 128, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(128, 256, 3, padding=1),      nn.ReLU(), nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1,2)        # [B,feat,T]
        x = self.conv(x)            # [B,256,T]
        x = self.pool(x).squeeze(-1)  # [B,256]
        return self.fc(x)

class AttentionClassifier(nn.Module):
    def __init__(self, feat_dim, attn_dim, num_classes, dropout=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        scores = self.attn(x)                  # [B,T,1]
        alpha  = F.softmax(scores, dim=1)      # [B,T,1]
        ctx    = (alpha * x).sum(1)            # [B,feat]
        ctx    = self.dropout(ctx)
        return self.fc(ctx)