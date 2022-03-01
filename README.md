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
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/df380a19-234e-4a2d-9da8-a562b1e15ab9/Untitled.png)
    
    - klue-nli-v1.1_dev.json DataFrame  , Dacon Test 와 결합 후 중복문장 제거한다음 test 부분만 잘라서 train에 결합시키기
    
    [train_dev.csv](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f09b7c38-32b8-41e2-9ea6-5b29531d51a8/train_dev.csv)
    
    [full_train.csv](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/759345ac-633b-4cdd-8ed2-6c51471b5804/full_train.csv)
    
    - full_train의 경우 578430 row ..too many man...
