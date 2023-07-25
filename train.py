import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import BertTokenizerFast, BertModel
from torch.utils.data import Dataset

from sklearn.metrics import f1_score

data_dir = 'data_nia'

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
        self.text = [e.split('\t')[0] for e in open(txt_file, 'r', encoding='utf8').readlines()]
        self.classtype = [e.split('\t')[1] for e in open(txt_file, 'r', encoding='utf8').readlines()]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        classtype = int(float(self.classtype[idx]))

        return text, classtype

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data 준비
dataset = NewsDataset(os.path.join(data_dir, 'train.txt'))
data_loader = DataLoader(dataset, 32, shuffle=True)

# 모델 정의
model_cls = BertBaseModel()
model_cls.to(device)
model_cls.train()

criterion = nn.CrossEntropyLoss(label_smoothing=0)
optimizer_cls = Adam(model_cls.parameters(), lr=5e-5)

#학습 진행
epoch = 3
running_loss_cls = 0.0

for e in range(epoch):
    for i, d in tqdm(enumerate(data_loader), desc='{}/{} epoch'.format(e+1, epoch), total=len(data_loader)):
        text, classtype = d
        optimizer_cls.zero_grad()

        text = list(text)
        input = tokenizer_bert(text, return_tensors="pt", padding=True)
        outputs_cls = model_cls(input["input_ids"].to(device))

        loss_cls = criterion(outputs_cls, classtype.to(device))

        loss_cls.backward()
        optimizer_cls.step()

        running_loss_cls += loss_cls.item()

        if i % 10 == 9:
            print('___________________________')
            print("running_loss_cls: ", running_loss_cls)

            running_loss_cls = 0.0

    # save
    torch.save(model_cls.state_dict(), "./model_state_dict.pt")


#test 진행
count = 0
right_count = 0

dataset_test = NewsDataset(os.path.join(data_dir, 'test.txt'))
data_loader_test = DataLoader(dataset_test, 1, shuffle=True)
torch.save(model_cls, './model')
model_cls.eval()

src_list = []
tgt_list = []

for i, d in enumerate(data_loader_test):
    count += 1
    text, classtype = d

    text = list(text)
    input = tokenizer_bert(text, return_tensors="pt", padding=True)
    outputs_cls = model_cls(input["input_ids"].to(device))[0].cpu()

    if classtype == torch.argmax(outputs_cls):
        right_count += 1
    #print(classtype, torch.argmax(outputs_cls))

    p = classtype.tolist()[0]
    q = torch.argmax(outputs_cls).tolist()

    src_list.append(p)
    tgt_list.append(q)

print(right_count/count)
f_score = f1_score(src_list, tgt_list, average='macro')
print(f_score)