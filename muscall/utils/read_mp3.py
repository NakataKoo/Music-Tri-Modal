import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = "/home_lab/nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/A/A/L/TRAALAH128E078234A.mp3"
sample_rate = 16000
num_samples = 960000
audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

# 波形を表示
plt.figure(figsize=(14, 5))
plt.plot(audio)
plt.title('Waveform of the audio')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

'''
import numpy as np
audio = audio.reshape(1, -1)
x = num_samples - audio.shape[1]
padded_audio = np.pad(audio[0], ((0, x)))
audio = np.array([padded_audio])
print(audio[0])
print(audio.shape)
'''