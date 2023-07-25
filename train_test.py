#-*- coding: utf-8 -*-

import os
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import BertTokenizerFast, AutoModelForSequenceClassification
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import f1_score

label2idx = {'사실형': 0, '추론형': 1, '대화형': 2, '예측형': 3}
# cer2idx = {'불확실': 0, '확실': 1}
# pol2idx = {'미정': 0, '부정': 1, '긍정': 2}
# tense2idx = {'과거': 0, '현재': 1, '미래': 2}

res = open('res.txt', 'w')
data_dir = 'data_nia'

tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
model_cls = AutoModelForSequenceClassification.from_pretrained("kykim/bert-kor-base")

#Dataset 클래스 정의
class NewsDataset(Dataset):
    def __init__(self, txt_file):
        df = pd.read_csv(txt_file, delimiter='\t')

        self.text = []
        self.label = []
        for idx in df.index[:300]:  ## todo: test
            self.text.append(tokenizer_bert(df.loc[idx, 'text'], return_tensors="pt", padding=True))
            self.label.append(df.loc[idx, 'label'])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = int(float(self.label[idx]))

        return text, label

#학습 준비
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_cls.to(device)
model_cls.train()

# data
df = pd.read_csv(os.path.join(data_dir, 'train.txt'), delimiter='\t')

input_ids = []
label = []
for idx in df.index[:300]:  ## todo: test
    input_ids.append(tokenizer_bert(df.loc[idx, 'text'], return_tensors="pt", padding=True))
    label.append(df.loc[idx, 'label'])
train_data = TensorDataset(torch.tensor(input_ids), torch.tensor(label))
train_sampler = RandomSampler(train_data)
dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

# train
criterion = nn.CrossEntropyLoss(label_smoothing=0)
optimizer = Adam(model_cls.parameters(), lr=5e-5)

epoch = 3
running_loss_cls = 0.0

optimizer.zero_grad()

for e in range(epoch):
    total_loss = 0

    # train
    for step, batch in tqdm(enumerate(dataloader), desc='{}/{} epoch'.format(e+1, epoch), total=len(dataloader)):
        text, label = batch
        print(text)
        print(label)
        exit()
        outputs = model_cls(text.to(device), labels=label.to(device))  # Forward 수행

        loss = outputs[0]  # 로스 구함
        total_loss += loss.item()  # 총 로스 계산

        loss.backward()  # Backward 수행으로 그래디언트 계산
        torch.nn.utils.clip_grad_norm_(model_cls.parameters(), 1.0)  # 그래디언트 클리핑
        optimizer.step()  # 그래디언트를 통해 가중치 파라미터 업데이트

        model_cls.zero_grad()  # 그래디언트 초기화


        if step % 10 == 9:
            print('___________________________')
            print("running_loss_cls: ", running_loss_cls)

            running_loss_cls = 0.0

    # save
    torch.save(model_cls.state_dict(), "./model_state_dict.pt")

dataset_val = NewsDataset(os.path.join(data_dir, 'test.txt'))
dataloader_test = DataLoader(dataset_val, 1, shuffle=True)
print('test set: ', dataloader_test.__len__())

# validation
count = 0
right_count = 0
src_list = []
tgt_list = []

for _, d_test in tqdm(enumerate(dataloader_test), desc='validation', total=len(dataloader_test)):
    count += 1
    text, classtype = d_test

    text = list(text)
    input = tokenizer_bert(text, return_tensors="pt", padding=True)
    outputs_cls = model_cls(input["input_ids"].to(device))[0].cpu()

    if classtype == torch.argmax(outputs_cls):
        right_count += 1

    p = classtype.tolist()[0]
    q = torch.argmax(outputs_cls).tolist()

    src_list.append(p)
    tgt_list.append(q)

print(right_count / count)
f_score = f1_score(src_list, tgt_list, average='macro')
print(f_score)