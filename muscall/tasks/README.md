# 性能評価

- Crosss-Modal Retrieval
- Zero-Shot Transfer Learning

## Cross-Modal Retrieval

クロスモーダル検索では、テキストから音楽を検索したり、音楽からテキストを検索したりする能力を評価。モデルがどれだけ正確に、テキストと対応する音楽やMIDI、その逆の対応関係を見つけられるかを測定。

具体的には、以下の指標が使用されます。

- R@K（Recall at K）: 上位K件の検索結果の中に正解が含まれている割合を示します。例えば、R@1は一番目の検索結果が正解である確率、R@10は上位10件に正解が含まれている確率。
- mAP10（Mean Average Precision at 10）: 上位10件の検索結果の中で、正解がどの位置にいるかを評価し、平均的な精度を測定。
- MedR（Median Rank）: 正解の検索結果がリストの何番目に表示されるかの中央値を示します。数値が小さいほど良い結果。

## Zero-Shot Transfer Learning

ゼロショット転移学習では、モデルが学習時には見たことのないタスクに対して、どれだけ上手に対応できるかを評価。元論文では、音楽ジャンル分類や自動タグ付けといったタスクが使用されている。

具体的には以下の指標が使用されます。

- Accuracy（精度）: 正解とモデルの予測が一致する割合。
- ROC-AUC（Receiver Operating Characteristic - Area Under Curve）: さまざまな閾値でのモデルの性能を評価し、真陽性率と偽陽性率の関係を示します。値が高いほどモデルが優れていることを示します。
- PR-AUC（Precision-Recall - Area Under Curve）: 再現率と適合率の関係を示し、クラスの不均衡がある場合に有効。