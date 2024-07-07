import pandas as pd
import os

# 读取 CSV 文件
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_entity = os.path.join(script_dir, 'entity_id.csv')
file_path_relation = os.path.join(script_dir, 'relation_id.csv')
file_path_triples = os.path.join(script_dir, 'softlogic.tsv')

entity_df = pd.read_csv(file_path_entity)
relation_df = pd.read_csv(file_path_relation)
triples_df = pd.read_csv(file_path_triples, sep='\t', header=None, names=['head', 'relation', 'tail', 'score'])


entity_to_id = dict(zip(entity_df['entity string'], entity_df['id']))
relation_to_id = dict(zip(relation_df['relation string'], relation_df['id']))


triples_df['head'] = triples_df['head'].map(entity_to_id)
triples_df['relation'] = triples_df['relation'].map(relation_to_id)
triples_df['tail'] = triples_df['tail'].map(entity_to_id)


if triples_df.isnull().values.any():
    print("Some entities or relations in train.tsv do not have corresponding IDs in entity_id.csv or relation_id.csv.")
    print(triples_df[triples_df.isnull().any(axis=1)])
else:
    # 保存处理后的数据
    triples_df.to_csv(file_path_triples, sep='\t', header=False, index=False)
    print("Processed triples saved to train.tsv")
