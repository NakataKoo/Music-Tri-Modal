import os
os.environ['OPENAI_API_KEY'] ="OPENAI_API_KEY"

from tqdm import tqdm
import json
import time

from openai import OpenAI
client = OpenAI()

# JSONファイルを読み込む
file_path = "dataset_all_short_caption_2_input.json"
output_path = "dataset_all_short_caption_2.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# `tqdm`を使用して進捗バーを表示しながらループを実行
for idx, item in enumerate(tqdm(data, desc="Processing captions"), start=1):
    time.sleep(1)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """
You are an excellent writer.
Summarize the English text into short sentences.
"""
            },
            {
                "role": "user",
                "content": """
# Steps

1. reading: fully comprehend the English text provided.
2. Extraction: Identify main ideas, key points, or conclusions.
3. summarizing: Summarize in short, effective sentences, including only key points. Omit unimportant details.

# Prior information

The input text to be summarized is a description of music.

# Output Format

- The post-summary should consist of 1 short sentences.
- The content and intent of the original text should be kept in mind and conveyed in a concise manner.
- The summary results should be written in a way that a novice musician can understand.
- No more than 15 words.

# Examples

- **Input:** "The quick brown fox jumps over the lazy dog. The dog doesn't seem bothered by the fox at all, remaining calm."
- **Output:** "A fox jumps over a calm dog."

# Input:
"""+f"""
{item['caption']}

# Output:
"""
            }
        ]
    )
    item['caption'] = completion.choices[0].message.content

    if idx % 1000 == 0:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(data[:idx], output_file, indent=4, ensure_ascii=False)

with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, indent=4, ensure_ascii=False)