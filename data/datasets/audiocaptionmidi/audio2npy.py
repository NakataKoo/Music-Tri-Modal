import librosa
import numpy as np
import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def save_audio_as_npy(audio_path):
    sample_rate = 16000
    duration = 45

    try:
        # 出力パスを生成
        npy_file_path = audio_path.replace('.mp3', '.npy')
        
        # オーディオデータを読み込み
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration, offset=0.1)
        audio = audio.reshape(1, -1)

        """
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

        clipped_audio = np.clip(audio[0], -1.0, 1.0)
        clipped_audio = 2 * (clipped_audio - np.min(clipped_audio)) / (np.max(clipped_audio) - np.min(clipped_audio)) - 1
        audio = np.array([clipped_audio])
        """
        # npyファイルとして保存
        np.save(npy_file_path, audio)
        
        # 処理が完了したMP3ファイルを削除
        os.remove(audio_path)
        
        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False

# MP3ファイルを再帰的に検索
data = glob.glob('/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/**/*.mp3', recursive=True)

# 進捗バーを表示しながら並列処理
with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(save_audio_as_npy, data), total=len(data), disable=False, leave=True))

# 結果の集計
success_count = sum(results)
print(f"Successfully processed {success_count} files out of {len(data)}")
