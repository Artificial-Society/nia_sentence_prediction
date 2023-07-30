import classifier

# train
CLS = classifier.Classification(model_name='kykim/bert-kor-base', min_sentence_length=5, MAX_LEN=64, batch_size=32, use_bert_tokenizer=True)
CLS.dataset(data_path='data/data_sample.tsv', col_sentence='text', col_label='label')
CLS.load_model(mode='train')
CLS.train(epochs=2, dataset_split=0.1)
