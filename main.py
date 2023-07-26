import classifier

# {'사실형': 0, '추론형': 1, '대화형': 2, '예측형': 3}

# train
CLS = classifier.Classification(model_name='kykim/bert-kor-base', min_sentence_length=5, MAX_LEN=256, batch_size=32, use_bert_tokenizer=True)
CLS.dataset(data_path='data/data.tsv', col_sentence='text', col_label='label')
CLS.load_model(mode='train')
CLS.train(epochs=3, dataset_split=0.1)

# inference
sentences = ['내일은 비가 올 것으로 예상됩니다.']
saved_model_path='model/saved/3'

CLS = classifier.Classification(model_name='kykim/bert-kor-base', min_sentence_length=5, MAX_LEN=256, batch_size=32, use_bert_tokenizer=True)
CLS.load_model(mode='inference', saved_model_path=saved_model_path)
logit = CLS.inference(sentences=sentences)
print(logit)