#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import BertTokenizerFast, BertModel
from torch.utils.data import Dataset

from sklearn.metrics import f1_score

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
model_state_dict = torch.load("./model_state_dict.pt", map_location=device)
model_cls.load_state_dict(model_state_dict)

model_cls.to(device)
model_cls.eval()

#test 진행
count = 0
right_count = 0

dataset_test = NewsDataset('data_nia/test.txt', 'data_nia/test_cer.txt', 'data_nia/test_pol.txt', 'data_nia/test_tense.txt')
data_loader_test = DataLoader(dataset_test, 1, shuffle=True)
torch.save(model_cls, './model')
model_cls.eval()

src_list = []
tgt_list = []

for i, d in enumerate(data_loader_test):
    count += 1
    text, classtype, certype, poltype, tensetype = d

    text = list(text)
    certype = certype.unsqueeze(1)
    poltype = poltype.unsqueeze(1)
    tensetype = tensetype.unsqueeze(1)

    inp_plus = torch.cat((certype, poltype, tensetype), -1).to(device)
    input = tokenizer_bert(text, return_tensors="pt", padding=True)
    outputs_cls = model_cls(input["input_ids"].to(device), inp_plus)[0].cpu()

    if classtype == torch.argmax(outputs_cls):
        right_count += 1
    #print(classtype, torch.argmax(outputs_cls))

    p = classtype.tolist()[0]
    q = torch.argmax(outputs_cls).tolist()

    res.write('정답: ' + str(p) + ', 결과: ' + str(q) + '\n')

    src_list.append(p)
    tgt_list.append(q)

print(right_count/count)
f_score = f1_score(src_list, tgt_list, average='macro')
print(f_score)