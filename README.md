# Dacon KorNLI 
## 월간데이콘 한국어 문장 분류 경진 대회[https://dacon.io/competitions/official/235875/overview/description] 
**1. train_data.csv**

├ Index : train data index

├ Premise : 실제 Text

├ Hypothesis : 가설 Text

└ Label : **참(Entailment)** 또는 **거짓(Contradiction)** 또는 **중립(Neutral)**

### **2. test_data.csv**

├ Index : test data index

├ Premise : 실제 Text

├Hypothesis : 가설 Text

└Label : 추론해야 하는 Label 값

#### Data

![Data](https://user-images.githubusercontent.com/76906638/156130464-44487242-1e98-4eb3-8c40-ace6a9ca9ed2.png)

- 두 문장 비교로 실제와 가설이 같으면 참 다를경우 거짓 또는 중립으로 label이 이루어짐

- Train Set 24,998 Test Set 1,666 (전체 60%) 개며 종료 전까진 1,666개로 얼마나 잘 맞췄는지 Accuracy 기반으로 채점 이후 종료 후  Full Test set으로 평가
    - 따라서 현재 Test Set 기반으로 Overfitting 만 조심하면 될듯..?
    - [https://dacon.io/competitions/official/235875/talkboard/405971?page=1&dtype=recent](https://dacon.io/competitions/official/235875/talkboard/405971?page=1&dtype=recent)
    - 위 링크는 KLUE Official 관련 링크이며 Dacon Test Set vs KLUE Official Dev Set => 3000개 中 6개만 일치
    - KLUE Offical Dev Set에서 Dacon Test Set 중복만 제거하고 사용가능하다는 답변을 받음 따라서 Train Dataset 에 추가하여 학습진행 가능
    - [https://klue-benchmark.com/tasks/68/data/download](https://klue-benchmark.com/tasks/68/data/download)
    
    ![KLUE](https://user-images.githubusercontent.com/76906638/156130801-f709a443-73d8-4c04-83da-227d1b7a83f4.png)
    
    - klue-nli-v1.1_dev.json DataFrame  , Dacon Test 와 결합 후 중복문장 제거한다음 test 부분만 잘라서 train에 결합시키기
    
    [train_dev.csv](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f09b7c38-32b8-41e2-9ea6-5b29531d51a8/train_dev.csv)
    
    [full_train.csv](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/759345ac-633b-4cdd-8ed2-6c51471b5804/full_train.csv)
    
    - full_train의 경우 578430 row ..too many man...
## Preprocess (데이터 전처리관련)

- 문장별 최소 최대 그리고 평균길이 관련한 EDA
    - [https://dacon.io/competitions/official/235875/codeshare/4387?page=1&dtype=recent](https://dacon.io/competitions/official/235875/codeshare/4387?page=1&dtype=recent)



- ~~KOELECTRA ( BATCH SIZE = 128 , EPOCH = 4 , Warm_up_ratio =1) 기준 비교 결과 특수문자 제거하고 돌리니까 성능이 미약하게 상승함~~
    - ~~하지만 저거 이후로 지금 성능을 못내고있음 ..~~
- 향후 전처리 관련 내용 더 추가 ..
- KorNLI KorSTS (추가 데이터 활용 방안)


###Run 
    python3 main.py

### 성능을 높일 수 있는 방안은 무엇?

[모델을 앙상블한다] 

KoBERT, KoELECTRA, roBERTa-large, KoGPT 등 여러 모델의 결과들를 앙상블

- 경진대회 앙상블 기법
    
    ![Ang](https://user-images.githubusercontent.com/76906638/156131164-899ead9f-cd48-487f-8820-d423cc25b685.png)
    

[데이터를 늘린다]

- Back translation : 한국어 → 영어 → 한국어
    
    파파고→ 5000개 기준 3시간 소요
    
    [크롤링 되지 않는 문장들 & 번역 후 premise = hypothesis인것들 다 drop시킴](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/96af2e8e-5aa3-495f-b6b9-fd38b6f1e61c/papago_train_final_drop_same.csv)
    
    크롤링 되지 않는 문장들 & 번역 후 premise = hypothesis인것들 다 drop시킴
    
    참고 : [https://dacon.io/competitions/official/235747/codeshare/3054?page=1&dtype=recent](https://dacon.io/competitions/official/235747/codeshare/3054?page=1&dtype=recent)
    
- Kakao brain에서 공개한 KorNLI 데이터셋 추가 (55만 개)
    
    [KorNLUDatasets/KorNLI at master · kakaobrain/KorNLUDatasets](https://github.com/kakaobrain/KorNLUDatasets/tree/master/KorNLI)
    
    여기서 이 대회 train 문장들과 유사도가 높은 문장들만 추려서 사용하는것은 어떤지?
    
    [Google Colaboratory](https://colab.research.google.com/github/Huffon/klue-transformers-tutorial/blob/master/sentence_transformers.ipynb#scrollTo=cad90326-2557-4b86-a42a-ca5033a09cfd)
    
    - Sentence BERT Code

[하이퍼 파라미터를 조정한다]

Tokenizer 길이 205(70으로 되어있음)으로 맞췄는지?

- Text_sum Length
    
    ![Length](https://user-images.githubusercontent.com/76906638/156131236-6c33879d-6d44-42d4-9cf0-904f764fa4da.png)
    
- Learning rate , Optimizer , loss
