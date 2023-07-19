############################################################################
### For 문 전에 위의 device, model, bert_model, tokenizer 를 선언해야 
### 매 호출 시 마다 새로 build 하지 않고 바로 사용이 가능함
############################################################################

############################################################################
############################################################################
## 아래는 실제 앱 내의 DB에 접근 불가능해 아래에 !SUDO CODE! 처럼 대략적인 작동 과정만 작성 ##
## 아래는 실제 앱 내의 DB에 접근 불가능해 아래에 !SUDO CODE! 처럼 대략적인 작동 과정만 작성 ##
## 아래는 실제 앱 내의 DB에 접근 불가능해 아래에 !SUDO CODE! 처럼 대략적인 작동 과정만 작성 ##
############################################################################
############################################################################

# 경로는 위치에 맞게 적절히 import
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer

from .summary.extract_keyword import create_stopword_set
from .summary.extract_keyword import get_keywords_top5
from .summary.create_summary import create_summary
from .summary.create_total_summary import create_total_summary

# 경로는 위치에 맞게 적절히 import
# 사전 정의된 불용어 목록 txt 파일 load
stopword_path = "./summary/data/stopwords-ko.txt"

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로는 위치에 맞게 적절히 import
# 사전 학습된 모델 load
model_path = "./summary/model/model.pt"
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.to(device)
model.eval()

# 사전 학습된 BERT model load
bert_model = SentenceTransformer('jhgan/ko-sbert-sts')
bert_model.to(device)

# 사전 학습된 tokenizer load
tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')

############################################################################################################################################

# Top5_News_Ticker DB 에서 요약문 생성

# 실제로는 앱 내의 DB load
Top5_News_Ticker = None

# 불용어 Class 생성
stopwords = create_stopword_set(stopword_path)

# 앱 내의 존재하는 DB에서 요약문 생성
for data in Top5_News_Ticker:
    # 실제로는 앱 내의 DB 의 field 읽음
    raw_text = data['raw_text']
    
    # 키워드 추출
    keyword = get_keywords_top5(raw_text, stopwords)
    
    # 실제로는 앱 내의 DB 의 field에 저장
    data['keyword'] = keyword
    
    # 요약문 생성
    summary = create_summary(raw_text, keyword.split(), model, tokenizer, device)
    
    # 실제로는 앱 내의 DB 의 field에 저장
    data['summary'] = summary
    
############################################################################################################################################

# 최종 Ticker_Summary DB 생성

# 실제로는 앱 내의 DB load
Top5_News_Ticker = None
Ticker_Summary = None

# 실제로는 앱 내의 DB 에 존재하는 ticker 번호 사용
ticker_list = None

# 실제로는 앱 내의 DB 로 ticker 별로 묶음 load
for ticker in ticker_list:
    
    # input 으로 사용할 list 선언
    summary_list = []
    
    # 실제로는 앱 내의 DB 로 append
    for data in Top5_News_Ticker:
        if data['ticker'] == ticker:
            summary_list.append(data['summary'])
    
    # 최종 요약문 생성
    total_summary = create_total_summary(summary_list, bert_model, model, tokenizer)
    
    # 실제로는 앱 내의 DB 의 field에 저장
    Ticker_Summary['total_summary'] = total_summary

