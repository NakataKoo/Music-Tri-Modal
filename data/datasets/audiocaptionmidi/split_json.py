import json
import random
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--input_file', type=str, default='data_creation/prepare_data/dict/CP.pkl')

    args = parser.parse_args()

    return args

def split_data(input_file, train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1):
    # JSONファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # データをシャッフル
    random.shuffle(data)
    
    # データの総数を取得
    total_count = len(data)
    
    # 各データセットのサイズを計算
    train_size = int(total_count * train_ratio)
    validation_size = int(total_count * validation_ratio)
    
    # データを分割
    train_data = data[:train_size]
    validation_data = data[train_size:train_size + validation_size]
    test_data = data[train_size + validation_size:]
    
    # 結果をファイルに書き出し
    with open('dataset_train.json', 'w', encoding='utf-8') as train_file:
        json.dump(train_data, train_file, ensure_ascii=False, indent=4)
    
    with open('dataset_val.json', 'w', encoding='utf-8') as validation_file:
        json.dump(validation_data, validation_file, ensure_ascii=False, indent=4)
    
    with open('dataset_test.json', 'w', encoding='utf-8') as test_file:
        json.dump(test_data, test_file, ensure_ascii=False, indent=4)

# 使用例
args = get_args()
input_file = args.input_file  # 統合したJSONファイルのパス
split_data(input_file)
