#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import BertTokenizerFast, BertModel
from torch.utils.data import Dataset

from tqdm import tqdm

res = open('res.txt', 'w')

#model 클래스 정의
class BertBaseModel(nn.Module):
    def __init__(self):
        super(BertBaseModel, self).__init__()
        self.bert = BertModel.from_pretrained("kykim/bert-kor-base")
        self.cls = nn.Linear(771, 4)
    def forward(self, data, plus):
        out = self.cls(torch.cat((self.bert(data)[1], plus), -1))

        return(out)


#Dataset 클래스 정의
class NewsDataset(Dataset):
    def __init__(self, txt_file, cer_file, pol_file, tense_file):
        self.text = [e.split('\t')[0] for e in open(txt_file, 'r', encoding='utf8').readlines()]
        self.classtype = [e.split('\t')[1] for e in open(txt_file, 'r', encoding='utf8').readlines()]
        self.certype = [e.split('\t')[1].strip('\n') for e in open(cer_file, 'r', encoding='utf8').readlines()]
        self.poltype = [e.split('\t')[1].strip('\n') for e in open(pol_file, 'r', encoding='utf8').readlines()]
        self.tensetype = [e.split('\t')[1].strip('\n') for e in open(tense_file, 'r', encoding='utf8').readlines()]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        classtype = int(float(self.classtype[idx]))
        certype = int(float(self.certype[idx]))
        poltype = int(float(self.poltype[idx]))
        tensetype = int(float(self.tensetype[idx]))

        return text, classtype, certype, poltype, tensetype

#학습 준비
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")

model_cls = BertBaseModel()
model_cls.to(device)
model_cls.train()

dataset = NewsDataset('data_nia/train.txt', 'data_nia/train_cer.txt', 'data_nia/train_pol.txt', 'data_nia/train_tense.txt')
print('train set: ', dataset.__len__())
dataset_a = NewsDataset('data_nia/val.txt', 'data_nia/val_cer.txt', 'data_nia/val_pol.txt', 'data_nia/val_tense.txt')
print('val set: ', dataset_a.__len__())
dataset_a = NewsDataset('data_nia/test.txt', 'data_nia/test_cer.txt', 'data_nia/test_pol.txt', 'data_nia/test_tense.txt')
print('test set: ', dataset_a.__len__())

data_loader = DataLoader(dataset, 32, shuffle=True)

criterion = nn.CrossEntropyLoss(label_smoothing=0)
optimizer_cls = Adam(model_cls.parameters(), lr=5e-5)

epoch = 3
running_loss_cls = 0.0

#학습 진행
for e in range(epoch):
    for i, d in tqdm(enumerate(data_loader), desc='{}/{} epoch'.format(e+1, epoch), total=len(data_loader)):
        text, classtype, certype, poltype, tensetype = d
        optimizer_cls.zero_grad()

        certype = certype.unsqueeze(1)
        poltype = poltype.unsqueeze(1)
        tensetype = tensetype.unsqueeze(1)

        inp_plus = torch.cat((certype, poltype, tensetype), -1).to(device)

        text = list(text)
        input = tokenizer_bert(text, return_tensors="pt", padding=True)
        outputs_cls = model_cls(input["input_ids"].to(device), inp_plus)

        loss_cls = criterion(outputs_cls, classtype.to(device))

        loss_cls.backward()
        optimizer_cls.step()

        running_loss_cls += loss_cls.item()

        if i % 10 == 9:
            print('___________________________')
            print("running_loss_cls: ", running_loss_cls)

            running_loss_cls = 0.0

torch.save(model_cls.state_dict(), "./model_state_dict.pt")