import os
import json
import pandas as pd
from tqdm import tqdm


# label2idx = {'사실형': 0, '추론형': 1, '대화형': 2, '예측형': 3}
# cer2idx = {'불확실': 0, '확실': 1}
# pol2idx = {'미정': 0, '부정': 1, '긍정': 2}
# tense2idx = {'과거': 0, '현재': 1, '미래': 2}
#
#
# dataset = []
# for file in tqdm(os.listdir('data_nia')):
#     if not file.endswith('json'):
#         continue
#
#     with open(os.path.join('data_nia', file), 'r', encoding='utf8') as file:
#         file_data = json.load(file)
#
#     for sentence in file_data['annotation']:
#         text = sentence['text']
#         label = label2idx[sentence['label']]
#         cer = cer2idx[sentence['value']['확실성']]
#         pol = pol2idx[sentence['value']['극성']]
#         tense = tense2idx[sentence['value']['시제']]
#
#         dataset.append({'text': text, 'label': label, 'cer': cer, 'pol': pol, 'tense': tense})
#
# df = pd.DataFrame(dataset)
# df.to_csv('data.txt', sep='\t',)

#########
# df = pd.read_csv('data_nia/data.txt', delimiter='\t')
# length = int(0.9*len(df))
#
# shuffled_df = df.sample(frac=1, random_state=42)
# train = shuffled_df[:length]
# test = shuffled_df[length:]
#
# print(len(train), len(test))
#
# train.to_csv('train.txt', sep='\t')
# test.to_csv('test.txt', sep='\t')


df = pd.read_csv('data_nia/test.txt', delimiter='\t')
df = df[['text', 'label']]
df.to_csv('test.txt', sep='\t', index=False)




# #######
# import pandas as pd
#
# mode = 'train'
# column_names = ['sentence', 'score']
#
# df = pd.read_csv('data_nia/{}.txt'.format(mode), delimiter='\t', names=column_names)
# print(len(df))
# df_cer = pd.read_csv('data_nia/{}_cer.txt'.format(mode), delimiter='\t', names=column_names)
# df_pol = pd.read_csv('data_nia/{}_pol.txt'.format(mode), delimiter='\t', names=column_names)
# df_tense = pd.read_csv('data_nia/{}_tense.txt'.format(mode), delimiter='\t', names=column_names)
# print(len(df), len(df_cer), len(df_pol), len(df_tense))
#
#
# df = df.merge(df_cer, on='sentence', how='left', suffixes=('', '_cer'))
# df = df.merge(df_pol, on='sentence', how='left', suffixes=('', '_pol'))
# df = df.merge(df_tense, on='sentence', how='left', suffixes=('_', '_tense'))
#
# df.dropna(inplace=True)
# df.drop_duplicates(inplace=True)
#
# # print(df)
#
#
# exit()