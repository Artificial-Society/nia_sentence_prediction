# nia_sentence_prediction
작성일 2023.3.27.  
작성자 윤상원(아티피셜소사이어티).  

모델 개관.  
- BERT 언어모델을 활용한 문장유형 분류 모델    
- 사전학습은 "kykim/bert-kor-base" 을 사용함.    

모델 구조    
<img width="1366" alt="스크린샷 2023-03-27 오후 3 02 47" src="https://user-images.githubusercontent.com/85025584/227861671-84ca3603-c33f-4d9b-9fc3-be0f658c0d39.png">


Input     
- 문장(raw 텍스트), 극성, 확실성, 시제 (각 0 부터 시작하는 정수로 라벨링함. ex) 과거시제 -> 0).   
- 문장은 KoBERT LM 에 input. KoBERT LM output 인 768차원 텐서와 극성, 확실성, 시제 라벨을 concat 해 771차원 텐서로 가공한 후, Linear 계층에 투입함.    

Output
- 4차원 텐서를 output 으로 받고, softmax 계층을 거쳐 가장 높은 확률을 갖는 값을 출력함.   
   
Task.   
- 문장(시퀀스) 분류.   

Training dataset.  
- 총 67155 문장.    
- 사실형 23360문장, 대화형 29134문장, 추론형 8220문장, 예측형 6441문장.   

Hyperparameters.   
- Cross Entropy loss function.  
- Adam optimizer.   
- learning rate = 5e-5.    
- 1 epoch.    
- 32 batch.    

Evaluation Metric.    
- 라벨 비율을 통일하여 8:1:1로 Train/Val/Test set 분리하였음.      
- Accuracy(macro) 와 F1 score 로 각 평가.      
- Accuracy 0.9218489397188468 (약 92%, 목표치 95%) / F1 0.8789739607118283 (약 88%, 목표치 85%).    

License.   
- Apache 2.0
