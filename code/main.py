# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 23:12:48 2022

@author: 이현호
"""

import pandas as pd 
import random
import numpy as np 
import os
import re
import transformers
import torch
import torch.nn as nn
import warnings 
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AdamW,RobertaForSequenceClassification
from transformers.optimization import get_cosine_schedule_with_warmup

#%%
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
seed_everything(42)

#%% data load 
def load_data(train='train_data.csv', test='test_data.csv'):
    train = pd.read_csv('../data/' + train)
    test = pd.read_csv('../data/' + test)
    submission = pd.read_csv('../data/sample_submission.csv')
    train_1 =pd.read_csv('../data/train_dev.csv')
    train_2 =pd.read_csv('../data/new_df.csv')
    train = pd.concat([train,train_1,train_2], ignore_index=True)
    label_dict = {"entailment" : 0, "contradiction" : 1, "neutral" : 2}
    train['label'] = train['label'].map(label_dict)
    # train['premise'] = train['premise'].map(lambda x: re.sub('[-=+.,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', x))
    # train['hypothesis'] = train['hypothesis'].map(lambda x: re.sub('[-=+.,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '',x))
    # test['premise'] = test['premise'].map(lambda x: re.sub('[-=+.,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', x))
    # test['hypothesis'] = test['hypothesis'].map(lambda x: re.sub('[-=+.,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', x))
    return train, test, submission

def concat(df):
    df["premise_"] = "[CLS]" + df["premise"] + "[SEP]"
    df["hypothesis_"] = df["hypothesis"] + "[SEP]"
    df["text_sum"] = df.premise_ + "" + df.hypothesis_
    df = df[['text_sum','label']]
    return df

train, test, submission = load_data()
train.drop_duplicates(inplace = True)
train.reset_index(drop=True,inplace=True)
train, test  = concat(train), concat(test)  

#%% 
device = torch.device("cuda")
EPOCHS = 6
batch_size = 32
lr = 0.00001
warmup_ratio = 0.1
pretrain = "klue/roberta-large"

#%%data loader 

class CustomDataset(Dataset):
  
  def __init__(self,dataset,option):
    
    self.dataset = dataset 
    self.option = option
    self.tokenizer = AutoTokenizer.from_pretrained(pretrain)

  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    row = self.dataset.iloc[idx, 0:2].values
    text = row[0]
    #y = row[1]

    inputs = self.tokenizer(
        text, 
        return_tensors='pt',
        truncation=True,
        max_length=105,
        pad_to_max_length=True,
        add_special_tokens=False
        )
    
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]
    
    if self.option =='train':
        y =row[1]
        return input_ids,attention_mask,y

    return input_ids, attention_mask


#%% Cross validation 


skf = StratifiedKFold(n_splits = 5,shuffle=True,random_state=42)
folds=[]
for trn_idx,val_idx in skf.split(train['text_sum'],train['label']):
    folds.append((trn_idx,val_idx))
    
#%%

best_models = []

from tensorboardX import SummaryWriter 

writer = SummaryWriter(logdir='/daintlab/home/ddualab/hyunho/tensorboard') 

for i,fold in enumerate(range(4,5)):
    print('===============',i+1,'fold start===============')
    model = RobertaForSequenceClassification.from_pretrained(pretrain,num_labels=3).to(device)
    model=nn.DataParallel(model).to(device)
    optimizer = AdamW(model.parameters(),lr=lr,weight_decay=1e-4,correct_bias=False)
    
    
    train_idx = folds[fold][0]
    valid_idx = folds[fold][1]
    train_data = train.loc[train_idx]
    val_data = train.loc[valid_idx]
    train_dataset = CustomDataset(train_data,'train')
    valid_dataset = CustomDataset(val_data,'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    warmup_ratio = 0.1
    total_steps = len(train_loader) * EPOCHS
    warmup_step = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=1, num_training_steps=total_steps)
    #valid_loss_min = 0.4
    valid_acc_max = 0.3
    val_acc_list = []

    for epoch in range(EPOCHS):
        print('===============',epoch+1,'epoch start===============')
        batches = 0
        total_loss = 0.0
        correct = 0
        total =0
        model.train()
        
        for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            y_batch = y_batch.to(device)
            y_pred = model(input_ids_batch.to(device), attention_mask = attention_masks_batch.to(device))[0]
            loss = F.cross_entropy(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y_batch).sum()
            total += len(y_batch)
            batches += 1
            if batches % 100 == 0:
                print("Batch Loss: ", total_loss, "Accuracy: ", correct.float() / total)
            writer.add_scalar('train/train_loss',total_loss,epoch)
            writer.add_scalar('train/train_acc',correct,epoch)
        val_loss = []
        val_acc = []
        
        for input_ids_batch, attention_masks_batch, y_batch in tqdm(valid_loader):
            
            model.eval()
            with torch.no_grad():
                
                y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
                valid_loss = F.cross_entropy(y_pred,y_batch.to(device)).cpu().detach().numpy()

                preds = torch.argmax(y_pred,1)
                preds = preds.cpu().detach().numpy()
                y_batch = y_batch.cpu().detach().numpy()
                batch_acc = (preds==y_batch).mean()
                val_loss.append(valid_loss)
                val_acc.append(batch_acc)
                
                
        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)
        val_acc_list.append(val_acc)
        scheduler.step()
        print(f'Epoch: {epoch}- valid Loss: {val_loss:.6f} - valid_acc : {val_acc:.6f}')
        print(optimizer.param_groups[0]["lr"])
        writer.add_scalar('val/val_loss',val_loss,epoch)
        writer.add_scalar('val/val_acc',val_acc,epoch)
        if max(val_acc_list) <= val_acc:
            torch.save(model.state_dict(), f'./models/Roberta_large_fold_4_{val_acc}.pth') 
            print('model save, model val acc : ',val_acc)
            print('best_models size : ',len(best_models))

    
#%%

test_dataset = CustomDataset(test,'test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

preds = []

for i in range(4): 
    print(f'fold{i} Start')
    # load my model
    #model = torch.load('./models/Roberta_large_modelpapago_2epoch.pt')
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(f'./models/best/Roberta_large_fold_{i}.pth'))
    model.eval()
    answer = []
    with torch.no_grad():
        for input_ids_batch, attention_masks_batch in tqdm(test_loader):
            y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0].detach().cpu().numpy()
            answer.extend(y_pred.argmax(axis=1))
    preds.append(answer)


np_pred = np.array(preds).T

pred = []
for i in range(1666):
    cnt = Counter(np_pred[i])
    pred.append(cnt.most_common()[0][0])
    
submission['label']=pred
label_dict1 = {0:"entailment" , 1: "contradiction" , 2:"neutral"}
submission['label']=submission['label'].map(label_dict1)
submission.to_csv('./predict/NLU_Roberta_Large_real_final_model.csv',index=False)