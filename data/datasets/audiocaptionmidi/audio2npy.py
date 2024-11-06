import librosa
import numpy as np
import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def save_audio_as_npy(audio_path):
    try:
        # 出力パスを生成
        npy_file_path = audio_path.replace('.mp3', '.npy')
        
        # オーディオデータを読み込み
        audio, sr = librosa.load(audio_path, sr=48000, mono=True, duration=45, offset=0.1)
        audio = audio.reshape(1, -1)
        
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
