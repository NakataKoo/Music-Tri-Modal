import librosa
import numpy as np

sample_rate = 16000
duration = 45
audio_path = "/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/A/A/A/TRAAAGR128F425B14B.mp3"

audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration, offset=0.1)
audio = audio.reshape(1, -1)

print(audio)
print(audio.shape)

num_samples = sample_rate * duration

# 短い音声に対しパディング
if audio.shape[1] < num_samples:
    x = num_samples - audio.shape[1]
    padded_audio = np.pad(audio[0], ((0, x)), "mean")
    audio = np.array([padded_audio])
# 長い音声に対しクロップ
elif audio.shape[1] > num_samples:
    cropped_audio = audio[0][:num_samples]
    audio = np.array([cropped_audio])

print(audio)
print(audio.shape)

clipped_audio = np.clip(audio[0], -1.0, 1.0)
clipped_audio = 2 * (clipped_audio - np.min(clipped_audio)) / (np.max(clipped_audio) - np.min(clipped_audio)) - 1
audio = np.array([clipped_audio])

print(audio)
print(audio.shape)