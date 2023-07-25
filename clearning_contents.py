import pandas as pd

mode = 'val'
column_names = ['sentence', 'score']

df = pd.read_csv('data_nia/{}.txt'.format(mode), delimiter='\t', names=column_names)
df_cer = pd.read_csv('data_nia/{}_cer.txt'.format(mode), delimiter='\t', names=column_names)
df_pol = pd.read_csv('data_nia/{}_pol.txt'.format(mode), delimiter='\t', names=column_names)
df_tense = pd.read_csv('data_nia/{}_tense.txt'.format(mode), delimiter='\t', names=column_names)
print(len(df), len(df_cer), len(df_pol), len(df_tense))


df = df.merge(df_cer, on='sentence', how='left', suffixes=('', '_cer'))
df = df.merge(df_pol, on='sentence', how='left', suffixes=('', '_pol'))
df = df.merge(df_tense, on='sentence', how='left', suffixes=('_', '_tense'))

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(df)


exit()