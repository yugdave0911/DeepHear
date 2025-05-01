import torchaudio
import librosa
import pandas as pd
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor, BertTokenizerFast, Wav2Vec2Model, AutoModel
from .utils import map_phq_score_to_label

class ContrastiveDataset(Dataset):
    """
    Expects a DataFrame with columns:
      audio_chunk_path, transcript_chunk_path
    Returns dict with 'audio' Tensor and 'text' Tensor per example.
    """
    def __init__(self,
                 samples_df,
                 sample_rate: int = 16000,
                 device: torch.device = None):
        self.df = samples_df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")
        self.wav_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.text_tok = AutoTokenizer.from_pretrained("roberta-base")
        self.text_model = AutoModel.from_pretrained("roberta-base").to(self.device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # audio
        wav, _ = librosa.load(row.audio_chunk_path,
                              sr=self.sample_rate)
        audio_inputs = self.wav_proc(wav, sampling_rate=self.sample_rate,
                                     return_tensors="pt", padding=True)
        audio_feat = self.text_model(**audio_inputs.to(self.device)).last_hidden_state

        # text
        txt = open(row.transcript_chunk_path).read()
        txt_in = self.text_tok(txt, truncation=True, padding="max_length",
                               max_length=512, return_tensors="pt")
        with torch.no_grad():
            text_feat = self.text_model(**txt_in.to(self.device)).last_hidden_state

        return {
            "audio": audio_feat.squeeze(0),
            "text":  text_feat.squeeze(0)
        }

class PHQ8AudioDataset(Dataset):
    def __init__(self, audio_paths, processor, device, phq_csv_path, model_name_audio, sampling_rate):
        self.audio_paths = audio_paths
        self.processor = processor
        self.device = device
        self.sr = sampling_rate
        self.phq_df = pd.read_csv(phq_csv_path).astype({"Participant_ID": str})
        self.backbone = Wav2Vec2Model.from_pretrained(model_name_audio).to(device).eval()

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        pid = re.search(r"participant_(\d+)_audio_chunk", path).group(1)
        score = int(self.phq_df.loc[self.phq_df.Participant_ID==pid, "PHQ8_Score"].iloc[0])
        label = map_phq_score_to_label(score)
        audio, _ = librosa.load(path, sr=self.sr)
        inputs = self.processor(audio.squeeze(0), sampling_rate=self.sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            feats = self.backbone(**inputs).last_hidden_state.squeeze(0)
        return feats, torch.tensor(label, device=self.device)

class PHQ8TextDataset(Dataset):
    def __init__(self, transcript_paths, tokenizer, device, phq_csv_path, model_name_text):
        self.paths = transcript_paths
        self.tokenizer = tokenizer
        self.device = device
        self.phq_df = pd.read_csv(phq_csv_path).astype({"Participant_ID": str})
        self.text_model = AutoModel.from_pretrained(model_name_text).to(device).eval()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pid = re.search(r"participant_(\d+)_transcript_chunk", path).group(1)
        score = int(self.phq_df.loc[self.phq_df.Participant_ID==pid, "PHQ8_Score"].iloc[0])
        label = map_phq_score_to_label(score)
        df = pd.read_csv(path)
        text = " ".join(str(u) for u in df["Utterance"].fillna("<pad>").tolist())
        enc = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.text_model(**enc).last_hidden_state.mean(dim=1).squeeze(0)
        return feats, torch.tensor(label, device=self.device)