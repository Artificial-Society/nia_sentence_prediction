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

# tokenizer 정의
tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")

#model 클래스 정의
class BertBaseModel(nn.Module):
    def __init__(self):
        super(BertBaseModel, self).__init__()
        self.bert = BertModel.from_pretrained("kykim/bert-kor-base")
        self.cls = nn.Linear(768, 4)
    def forward(self, data):
        return self.cls(self.bert(data)[1])

# Dataset 클래스 정의
class NewsDataset(Dataset):
    def __init__(self, txt_file):

        self.input_ids = []
        self.label = []
        for e in open(txt_file, 'r', encoding='utf8').readlines()[1:]: # 첫줄은 column명
            self.input_ids.append(tokenizer_bert(e.split('\t')[0], return_tensors="pt", padding="max_length", truncation=True)['input_ids'])
            # self.input_ids.append(e.split('\t')[0])
            self.label.append(int(e.split('\t')[1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        text = self.input_ids[idx]
        label = self.label[idx]

        return text, label

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data 준비
dataset = NewsDataset(os.path.join(data_dir, 'test.txt'))
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# 모델 정의
model_cls = BertForSequenceClassification.from_pretrained("kykim/bert-kor-base") # BertBaseModel()
model_cls.to(device)
model_cls.train()

criterion = nn.CrossEntropyLoss(label_smoothing=0)
optimizer_cls = Adam(model_cls.parameters(), lr=5e-5)

#학습 진행
epoch = 3
running_loss_cls = 0.0

for e in range(epoch):
    for i, batch in tqdm(enumerate(dataloader), desc='{}/{} epoch'.format(e+1, epoch), total=len(dataloader)):
        optimizer_cls.zero_grad()

        batch = tuple(t.to(device) for t in batch)
        input_ids, label = batch

        outputs = model_cls(input_ids.reshape(batch_size, 512), labels=label)
        loss_cls = outputs[0]  # 로스 구함

        loss_cls.backward()
        optimizer_cls.step()

        running_loss_cls += loss_cls.item()

        if i % 100 == 0:
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
    input_ids, label = d

    with torch.no_grad():
        outputs = model_cls(input_ids.reshape(batch_size, 512))
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