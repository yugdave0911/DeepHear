import os
import pandas as pd
from pydub import AudioSegment

def chunk_participant_data_multiple(
    audio_file_paths: list[str],
    transcript_file_paths: list[str],
    output_audio_dir: str,
    output_transcript_dir: str,
    chunk_length_ms: int = 180_000
) -> pd.DataFrame:
    """
    For each session:
      - Load audio + transcript CSV (must have columns: speaker, start_time, stop_time, value)
      - Extract only Participant segments, concatenate them
      - Save combined session audio/transcript
      - Chunk combined audio into fixed-length pieces and save per-chunk transcript
    Returns a DataFrame with columns [session_id, chunk_index, audio_chunk_path, transcript_chunk_path].
    """
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_transcript_dir, exist_ok=True)
    records = []

    for session_id, (aud_fp, txt_fp) in enumerate(zip(audio_file_paths, transcript_file_paths)):
        # load transcript and audio
        df = pd.read_csv(txt_fp)
        parts = df[df["speaker"] == "Participant"].reset_index(drop=True)
        audio = AudioSegment.from_file(aud_fp)

        # build continuous Participant stream
        segs, new_rows, accum_ms = [], [], 0
        for _, row in parts.iterrows():
            start_ms = int(row["start_time"] * 1000)
            end_ms = int(row["stop_time"] * 1000)
            seg = audio[start_ms:end_ms]
            segs.append(seg)
            dur = len(seg)
            new_rows.append({
                "speaker": "Participant",
                "start_time": accum_ms / 1000,
                "stop_time": (accum_ms + dur) / 1000,
                "value": row.get("value", "")
            })
            accum_ms += dur

        combined = segs[0] if segs else AudioSegment.empty()
        for s in segs[1:]:
            combined += s

        # save combined session files
        sess_audio = os.path.join(output_audio_dir, f"session_{session_id}.wav")
        sess_txt   = os.path.join(output_transcript_dir, f"session_{session_id}.csv")
        combined.export(sess_audio, format="wav")
        pd.DataFrame(new_rows).to_csv(sess_txt, index=False)

        # now chunk
        n_chunks = (len(combined) + chunk_length_ms - 1) // chunk_length_ms
        for ci in range(n_chunks):
            start = ci * chunk_length_ms
            end   = min((ci+1)*chunk_length_ms, len(combined))
            chunk = combined[start:end]

            # filter transcript rows within this chunk
            chunk_rows = []
            for r in new_rows:
                rs = int(r["start_time"]*1000)
                re = int(r["stop_time"]*1000)
                if rs >= start and re <= end:
                    chunk_rows.append({
                        **r,
                        "start_time": (rs - start)/1000,
                        "stop_time":  (re - start)/1000
                    })

            audio_path = os.path.join(output_audio_dir,
                                      f"session_{session_id}_chunk_{ci}.wav")
            txt_path   = os.path.join(output_transcript_dir,
                                      f"session_{session_id}_chunk_{ci}.csv")
            chunk.export(audio_path, format="wav")
            pd.DataFrame(chunk_rows).to_csv(txt_path, index=False)

            records.append({
                "session_id": session_id,
                "chunk_index": ci,
                "audio_chunk_path": audio_path,
                "transcript_chunk_path": txt_path
            })

    return pd.DataFrame(records)