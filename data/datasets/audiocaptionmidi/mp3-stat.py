import os
from mutagen.mp3 import MP3
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import statistics
import numpy as np

def get_mp3_duration(file_path):
    """MP3ファイルの長さを取得する関数"""
    try:
        audio = MP3(file_path)
        return audio.info.length, file_path  # 長さとパスをタプルで返す
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_mp3_durations(directory, threshold):
    # 対象のmp3ファイルを収集
    mp3_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))

    # tqdmで進捗を表示しながら並列処理を実行
    durations = []
    with ProcessPoolExecutor() as executor:
        # ファイルごとに非同期にMP3の長さを取得
        futures = {executor.submit(get_mp3_duration, file): file for file in mp3_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing MP3 files"):
            result = future.result()
            if result is not None:
                durations.append(result)  # タプル (length, file_path) を追加

    if durations:
        # 各ファイルの長さだけのリストを作成
        lengths = [duration[0] for duration in durations]
        
        # 最大・最小ファイルの特定
        max_duration_file = max(durations, key=lambda x: x[0])
        min_duration_file = min(durations, key=lambda x: x[0])
        
        # 統計値の計算
        avg_duration = statistics.mean(lengths)
        median_duration = statistics.median(lengths)
        q1_duration = np.percentile(lengths, 25)  # 第1四分位数
        q3_duration = np.percentile(lengths, 75)  # 第3四分位数
        iqr = q3_duration - q1_duration  # 四分位範囲

        # 外れ値のしきい値計算
        lower_bound = q1_duration - 1.5 * iqr
        upper_bound = q3_duration + 1.5 * iqr
        
        # 外れ値の特定
        outliers = [(duration, path) for duration, path in durations if duration < lower_bound or duration > upper_bound]

        # 指定した秒数を超えるファイルの数をカウント
        above_threshold = [(duration, path) for duration, path in durations if duration > threshold]

        # 結果を表示
        print(f"Total MP3 files found: {len(durations)}")
        print(f"Maximum duration: {max_duration_file[0]:.2f} seconds - File: {max_duration_file[1]}")
        print(f"Minimum duration: {min_duration_file[0]:.2f} seconds - File: {min_duration_file[1]}")
        print(f"Average duration: {avg_duration:.2f} seconds")
        print(f"Median duration: {median_duration:.2f} seconds")
        print(f"1st Quartile (Q1): {q1_duration:.2f} seconds")
        print(f"3rd Quartile (Q3): {q3_duration:.2f} seconds")
        print(f"IQR: {iqr:.2f} seconds")
        print(f"Lower bound for outliers: {lower_bound:.2f} seconds")
        print(f"Upper bound for outliers: {upper_bound:.2f} seconds")
        
        # 外れ値の表示
        if outliers:
            print("\nOutlier files (outside of IQR bounds):")
            for duration, path in outliers:
                print(f"Duration: {duration:.2f} seconds - File: {path}")
        else:
            print("\nNo outliers found based on the IQR method.")
        
        # 指定秒数を超えるファイルの数を表示
        print(f"\nNumber of files exceeding {threshold} seconds: {len(above_threshold)}")
        if above_threshold:
            print("Files exceeding the threshold:")
            for duration, path in above_threshold:
                print(f"Duration: {duration:.2f} seconds - File: {path}")
                pass
    else:
        print("No MP3 files found in the specified directory.")

# 使用例: 特定のディレクトリAを指定し、しきい値を設定
directory_a = '/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3'
threshold_seconds = 65  # しきい値（秒）
get_mp3_durations(directory_a, threshold_seconds)