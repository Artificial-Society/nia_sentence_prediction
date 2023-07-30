import classifier
from constants import label2idx

idx2label = {v: k for k, v in label2idx.items()}

# inference

saved_model_path='model/saved'

CLS = classifier.Classification(model_name='kykim/bert-kor-base', min_sentence_length=5, MAX_LEN=64, batch_size=32, use_bert_tokenizer=True)
CLS.load_model(mode='inference', saved_model_path=saved_model_path)


while True:
    print('예측할 문장을 입력해 주세요.')
    sentences = input()
    # sentences = ['밀레니얼은 부캐 열풍을 유쾌하게 받아들이고 있다.', '그런데 정기간행물은 잡지와 기타 간행물을 제외하면 거의 변동이 없었다.']

    sentences = sentences if isinstance(sentences, list) else [sentences]
    logits = CLS.inference(sentences=sentences if isinstance(sentences, list) else [sentences])

    print('----------------')
    for sentence, pred in zip(sentences, logits):
        print('{}: {}'.format(idx2label[pred], sentence))
    print('----------------')
