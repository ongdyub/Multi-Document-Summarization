import torch
import operator
from collections import defaultdict
from transformers import AutoTokenizer, BartForConditionalGeneration
from krwordrank.word import KRWordRank

# 하이퍼파라미터 설정
# GPU RAM 상황에 따라서 batch_size = 2 ~ 10
batch_size = 10
num_epochs = 50
learning_rate = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
input_max_length = 1024
output_max_length = 128

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')
    return tokenizer

# 긴 문장 전용 요약 함수
# 1. 기사 내 키워드에 대한 점수 dict 추출 함수
# Input : 'article : 긴 기사 원문 string'
# Output : '해당 기사의 keyword list'
def extract_keyword_scores(article, min_count=2, max_length=3, verbose=False, stopwords=[]):
    # 문장 단위로 나누기
    sentences = article.split(".")
    
    wordrank_extractor = KRWordRank(
        min_count = min_count,   # 단어의 최소 출현 빈도수
        max_length = max_length, # 단어의 최대 길이
        verbose = verbose
    )

    keywords, rank, graph = wordrank_extractor.extract(sentences)
    for stopword in stopwords:
        keywords.pop(stopword, None)
    
    return keywords

# 2. 키워드 점수 dict를 이용해 기사 내의 문장들에 대해 중요도 점수를 계산하는 함수
# Input : 'article : 긴 기사 원문 string, keyword_score : 1에서 생성된 키워드 스코어 list'
# Output : '각 문장의 score list'
def compute_sentence_scores(article, keyword_scores):
    sentences = article.split(".")
    sentence_scores = defaultdict(float)
    
    for sentence in sentences:
        for word, score in keyword_scores.items():
            if word in sentence:
                sentence_scores[sentence] += score

    return sentence_scores


# 3. 가장 낮은 점수를 가진 문장을 제거하면서 기사 전체 길이가 1024 이하가 되도록 만드는 함수
# Input : '긴 기사 원문 string', '각 문장의 score list'
# Output : '짧은 기사 원문 string'
def trim_article(article, sentence_scores, max_length=1024):
    sentences = article.split(".")
    tokenizer = get_tokenizer()
    
    while len(tokenizer.encode(article)) > max_length:
        # 가장 점수가 낮은 문장 찾기
        try:
            min_score_sentence = min(sentence_scores.items(), key=operator.itemgetter(1))[0]
            
            if min_score_sentence in sentences:
                sentences.remove(min_score_sentence)
                # 제거된 문장의 점수 정보도 제거
                sentence_scores.pop(min_score_sentence)
        except:
            sentences = sentences[:len(sentences)-2]
        
        # 기사 다시 조합
        article = '.'.join(sentences)
    
    return article

# 4. 긴 문장의 기사의 길이를 줄여주는 함수
# Input : '긴 기사 원문 string'
# Output : '짧은 기사 원문 string'
def summarize_article(article, max_length=1024, min_count=2, max_length_word=3, verbose=False, stopwords=[]):
    # 1. 기사 내 키워드에 대한 점수 dict 추출
    keyword_scores = extract_keyword_scores(article, min_count, max_length_word, verbose, stopwords=stopwords)
    
    # 2. 키워드 점수 dict를 이용해 기사 내의 문장들에 대해 중요도 점수를 계산
    sentence_scores = compute_sentence_scores(article, keyword_scores)
    
    # 3. 가장 낮은 점수를 가진 문장을 제거하면서 기사 전체 길이가 1024 이하가 되도록 만드는 과정
    summarized_article = trim_article(article, sentence_scores, max_length)
    
    return summarized_article


import sys
import torch
import numpy as np
from transformers import AutoTokenizer, BartForConditionalGeneration, AdamW
from transformers import get_cosine_schedule_with_warmup

# 모델, 토크나이저 가져옴
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
# 학습 데이터 오픈
import json

# 각 dataset 은 "content" 와 "summary" 를 가져야 하고
db_dataset_path = "./data/DB_train_Final.json"
opensource_dataset_path = "./data/OS_train_Final.json"

# db_dataset_path = "/content/drive/MyDrive/Tune/data/DB_train_Final.json"
# opensource_dataset_path = "/content/drive/MyDrive/Tune/data/OS_train_Final.json"

with open(db_dataset_path) as f:
    db_dataset = json.load(f)

with open(opensource_dataset_path) as f:
    opensource_dataset = json.load(f)

# 각 loss 별 가중치
keyword_loss_weight = 1
simple_loss_weight = 1

# 모델 GPU 로 이동
model.to(device)

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 토크나이즈 한 다음, dataLoader 로 변경
class ArticleDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        content = item["content"]
        summary = item["summary"]
        keywords = item["keywords"]
        
        # content 에 '키워드 : 키워드 1, 키워드 2, ..., 키워드 5' 추가
        keywords_prefix = '(키워드 : '
        for i, kw in enumerate(keywords):
            keywords_prefix += kw
            if i != len(keywords) - 1:
                keywords_prefix += ', '
            else:
                keywords_prefix += ")"
        
        content = keywords_prefix + content

        while len(keywords) < 5:
            keywords.append("")

        # 길이가 긴 데이터 처리
        if(len(self.tokenizer.encode(content)) > 1024):
            content = summarize_article(item["content"])
        
        inputs = self.tokenizer(content, truncation=True, padding="max_length", max_length=input_max_length, return_tensors="pt")
        targets = self.tokenizer(summary, truncation=True, padding="max_length", max_length=output_max_length, return_tensors="pt")

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = targets['input_ids'].squeeze()

        return input_ids, attention_mask, labels, keywords
    

# 불필요 텍스트 학습 시에 제거
# 광고, 언론사 정보 등이 이에 해당
must_remove_char = [";", "▶", "ⓒ", "©", "☞", "&", "★", "☆", "✩", "&nbsp"]
def remove_substrings(data_list, target_string):
    substrings = set()
    updated_list = []
    for data in data_list:
        s = data['content']
        if target_string in s:
            last_occurrence = s.rfind(target_string)
            substring = s[last_occurrence + len(target_string):]
            if substring and any(char in substring for char in must_remove_char):
                substrings.add(substring)
                s = s[:last_occurrence + len(target_string)]  # substring 제거
            data['content'] = s
        updated_list.append(data)
    return substrings, updated_list
_, db_dataset = remove_substrings(db_dataset, "다.")

db_train, db_test = train_test_split(db_dataset, test_size=0.1, random_state=42)
opensource_train, opensource_test = train_test_split(opensource_dataset, test_size=0.1, random_state=42)

db_train_dataset = ArticleDataset(db_train, tokenizer)
db_test_dataset = ArticleDataset(db_test, tokenizer)

db_train_loader = DataLoader(db_train_dataset, batch_size=batch_size, shuffle=False)
db_test_loader = DataLoader(db_test_dataset, batch_size=batch_size, shuffle=False)

opensource_train_dataset = ArticleDataset(opensource_train, tokenizer)
opensource_test_dataset = ArticleDataset(opensource_test, tokenizer)

opensource_train_loader = DataLoader(opensource_train_dataset, batch_size=batch_size, shuffle=False)
opensource_test_loader = DataLoader(opensource_test_dataset, batch_size=batch_size, shuffle=False)

# 학습 스케줄링 파라미터
total_steps = (len(db_train_loader) + len(opensource_train_loader)) * num_epochs
warmup_steps = int(total_steps * 0.1)

# Optimizer 및 Scheduler 설정
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, correct_bias=False)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


import torch.nn as nn
# 두 가지 loss function
cross_entropy = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

def simple_finetuning_loss(logits, labels):
    return cross_entropy(logits.view(-1, logits.shape[-1]), labels[:,1:].view(-1))

# keywords 에 들어가야 하는 것은 현재 batch 의 기사에 해당하는 키워드들
# keywords = [[기사 1 키워드 5개], [기사 2 키워드 5개], ...]
def keyword_loss_function(outputs, labels, attention_mask, keywords, tokenizer):
    generated_summary_ids = torch.argmax(outputs.logits, dim=-1)
    generated_summary = tokenizer.batch_decode(generated_summary_ids, skip_special_tokens=True)
    
    keyword_loss = 0
    # batch 내의 summary 순회
    for idx, summary in enumerate(generated_summary):
        keyword_count = sum([1 for keyword in keywords[idx] if keyword in summary])
        
        if(keyword_count == 5):
            keyword_loss += 0
        elif(keyword_count == 4):
            keyword_loss += 0
        elif(keyword_count == 3):
            keyword_loss += 0
        elif(keyword_count == 2):
            keyword_loss += 3
        elif(keyword_count == 1):
            keyword_loss += 15
        elif(keyword_count == 0):
            keyword_loss += 63

    keyword_loss /= len(generated_summary)
    
    return keyword_loss

# earlystopping 구현, 3 or 5 가 일반적으로 사용
class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # 모델 저장
            save_directory = f"./save/{self.counter}"
            # save_directory = f"./save/{self.counter}"

            # 모델의 파라미터 저장
            torch.save(model.state_dict(), f"{save_directory}/model.pt")
            
            if self.verbose:
                print(f"Validation loss decreased ({self.best_score:.4f} --> {score:.4f}).")
        elif score < self.best_score:
            self.counter += 1

            # 모델 저장
            save_directory = f"./save/{self.counter}"

            # 모델의 파라미터 저장
            torch.save(model.state_dict(), f"{save_directory}/model.pt")

            # tokenizer 저장
            tokenizer.save_pretrained(save_directory)
            
            if self.verbose:
                print(f"Patience counter: {self.counter} out of {self.patience}.")
            if self.counter >= self.patience:
                
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            # 모델 저장
            save_directory = f"./save/{self.counter}"

            # 모델의 파라미터 저장
            torch.save(model.state_dict(), f"{save_directory}/model.pt")

early_stopping = EarlyStopping(verbose=True)


# train 과 evaluate 정의
from tqdm import tqdm

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    total_simple_loss = 0
    total_keyword_loss = 0

    for i, batch in enumerate(tqdm(dataloader)):
        inputs, attention_mask, labels, keywords = batch
        # print("inputs:", inputs)  # Debugging line
        # print("attention_mask:", attention_mask)  # Debugging line
        # print("labels:", labels)  # Debugging line
        # print(keywords)
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
        logits = outputs.logits

        simple_loss = outputs.loss
        keyword_loss = keyword_loss_function(outputs, labels, attention_mask, list(zip(*keywords)), tokenizer)

        # print(simple_loss)
        # print(keyword_loss)

        # 전체 loss
        batch_loss = simple_loss * keyword_loss
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

        # 상태를 자세히 보기 위해 저장
        total_simple_loss += simple_loss.item()
        total_keyword_loss += keyword_loss
        scheduler.step()
        

    return total_loss / len(dataloader), total_simple_loss / len(dataloader), total_keyword_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_simple_loss = 0
    total_keyword_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, attention_mask, labels, keywords = batch

            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits

            simple_loss = outputs.loss
            keyword_loss = keyword_loss_function(outputs, labels, attention_mask, list(zip(*keywords)), tokenizer)

            # 전체 loss
            batch_loss = simple_loss_weight * simple_loss + keyword_loss_weight * keyword_loss
            total_loss += batch_loss.item()

            # 상태를 자세히 보기 위해 저장
            total_simple_loss += simple_loss.item()
            total_keyword_loss += keyword_loss

    return total_loss / len(dataloader), total_simple_loss / len(dataloader), total_keyword_loss / len(dataloader)


# 학습 코드
# Early Stop 이 출력 될 시 -> save/0/ 의 model0.pt 를 사용
# Early Stop 이 출력 되지 않고 모든 에폭에 대해 학습 완료 시 -> save 폴더의 model.pt 를 사용
for epoch in range(num_epochs):
    train_loss, s_loss, k_loss = train(model, opensource_train_loader, optimizer)
    eval_loss, e_s_loss, e_k_loss = evaluate(model, opensource_test_loader)

    print(f"Epoch: {epoch+1}, Opensource set")
    print(f"Training Loss: {train_loss:.4f}, Simple Loss: {s_loss:.4f}, Keyword loss : {k_loss:.4f}")
    print(f"Evaluation Loss: {eval_loss:.4f}, Simple Loss: {e_s_loss:.4f}, Keyword loss : {e_k_loss:.4f}")

    train_loss, s_loss, k_loss = train(model, db_train_loader, optimizer)
    eval_loss, e_s_loss, e_k_loss = evaluate(model, db_test_loader)

    print(f"Epoch: {epoch+1}, DB set")
    print(f"Training Loss: {train_loss:.4f}, Simple Loss: {s_loss:.4f}, Keyword loss : {k_loss:.4f}")
    print(f"Evaluation Loss: {eval_loss:.4f}, Simple Loss: {e_s_loss:.4f}, Keyword loss : {e_k_loss:.4f}")

    early_stopping(eval_loss, model)

    if early_stopping.early_stop:
        print("Early stopping.")
        break

# 모델 저장
save_directory = "./save"

# 모델의 파라미터 저장
torch.save(model.state_dict(), f"{save_directory}/model.pt")

# tokenizer 저장
tokenizer.save_pretrained(save_directory)
