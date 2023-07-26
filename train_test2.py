import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification
from torch.utils.data import Dataset

from sklearn.metrics import f1_score

data_dir = 'data_nia'
batch_size = 32
epoch = 3
lr = 5e-5

# tokenizer / model 정의
tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
model_cls = BertForSequenceClassification.from_pretrained("kykim/bert-kor-base", num_labels=4)

# Dataset 클래스 정의
class NewsDataset(Dataset):
    def __init__(self, txt_file):

        self.text = []
        self.label = []
        for e in open(txt_file, 'r', encoding='utf8').readlines()[1:]: # 첫줄은 column명
            # self.text.append(tokenizer_bert(e.split('\t')[0], return_tensors="pt", padding="max_length", truncation=True)['input_ids'])
            self.text.append(e.split('\t')[0])
            self.label.append(int(e.split('\t')[1]))

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]

        return text, label

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data 준비
dataset = NewsDataset(os.path.join(data_dir, 'test.txt'))  ## todo: test
dataloader = DataLoader(dataset, batch_size, shuffle=True)



#학습 진행
model_cls.to(device)
model_cls.train()

optimizer_cls = Adam(model_cls.parameters(), lr=lr)

running_loss_cls = 0.0

for e in range(epoch):
    for i, batch in tqdm(enumerate(dataloader), desc='{}/{} epoch'.format(e+1, epoch), total=len(dataloader)):
        optimizer_cls.zero_grad()
        text, label = batch

        input_ids = tokenizer_bert(text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        outputs = model_cls(input_ids['input_ids'].to(device), labels=label.to(device))
        loss_cls = outputs[0]  # 로스 구함

        loss_cls.backward()
        optimizer_cls.step()

        running_loss_cls += loss_cls.item()

        if i % 10 == 0:
            print('___________________________')
            print("running_loss_cls: ", running_loss_cls)

            running_loss_cls = 0.0

    # save
    torch.save(model_cls.state_dict(), "./model_state_dict.pt")


#test 진행
count = 0
right_count = 0
src_list = []
tgt_list = []

dataset_test = NewsDataset(os.path.join(data_dir, 'test.txt'))
dataloader_test = DataLoader(dataset_test, 1, shuffle=True)

model_cls.eval()

for i, d in enumerate(dataloader_test):
    count += 1
    text, label = d

    with torch.no_grad():
        input_ids = tokenizer_bert(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model_cls(input_ids['input_ids'].to(device))
    outputs_cls = outputs[0]

    if label == torch.argmax(outputs_cls):
        right_count += 1

    p = label.tolist()[0]
    q = torch.argmax(outputs_cls).tolist()

    src_list.append(p)
    tgt_list.append(q)

print(right_count/count)
f_score = f1_score(src_list, tgt_list, average='macro')
print(f_score)