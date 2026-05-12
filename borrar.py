import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
df = pd.read_csv('data/processed/train_cleaned.csv')
row = df.iloc[0]

prefix = 'traduce glosa LSM al español: '
inp = tokenizer(prefix + str(row['MSLG']), max_length=128, truncation=True, padding='max_length')
lab = tokenizer(str(row['SPA']), max_length=128, truncation=True, padding='max_length')

label_ids = [(l if l != tokenizer.pad_token_id else -100) for l in lab['input_ids']]

print('Input ids (primeros 10):', inp['input_ids'][:10])
print('Label ids (primeros 10):', label_ids[:10])
print('Labels no-padding:', sum(1 for l in label_ids if l != -100))
print('MSLG:', row['MSLG'])
print('SPA:', row['SPA'])
