# nia_sentence_prediction
DEMO: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://github.com/Artificial-Society/nia_sentence_prediction/blob/main/sentence_type_prediction.ipynb](https://colab.research.google.com/github/Artificial-Society/nia_sentence_prediction/blob/main/sentence_type_prediction.ipynb))

모델 개관.  
- BERT 언어모델을 활용한 문장유형 분류 모델    
- 사전학습 모델로 "kykim/bert-kor-base" 을 사용함.    

모델 구조    
<img width="1366" alt="스크린샷 2023-03-27 오후 3 02 47" src="https://user-images.githubusercontent.com/85025584/227861671-84ca3603-c33f-4d9b-9fc3-be0f658c0d39.png">

Hyperparameters.   
- learning rate = 5e-5.    
- 3 epoch.    
- 32 batch.    

Evaluation Metric.        
- Accuracy(macro) 와 F1 score

License.   
- Apache 2.0
