# Set-up Extract Keyword

import json
import time
from krwordrank.word import summarize_with_keywords
from konlpy.tag import Kkma
from konlpy.tag import Okt
import numpy as np
from tqdm import tqdm
import re

class SentenceTokenizer(object):
    def __init__(self, stopwords):
        self.kkma = Kkma()
        self.okt = Okt()
        self.stopwords = stopwords

    # sentences 단위로 구분 
    def text2sentences(self, text): 
        sentences = self.kkma.sentences(text)
        return sentences

    # sentences에서 명사를 추출 
    def get_nouns(self, sentences): 
        nouns = []
        for sentence in sentences:
            if sentence != '':
                nouns.append(' '.join([noun for noun in self.okt.nouns(str(sentence)) 
                  if noun not in self.stopwords and len(noun) > 1]))
        return nouns

def create_stopword_set(stopword_path):
  stopwords_new = set(line.strip() for line in open(stopword_path, "r", encoding="utf-8"))
  stopwords = {'위해', '지원', '이번', '시장', '있다', '서비스', '대한', '밝혔다', '기술', '있는', '위한', '것으로', '말했다', '올해', '사업', '국내', '기업', '있습니다', '20', '지난', '통해'} # 이전에 추출된 키워드 반영
  stopwords.update(stopwords_new)
  return stopwords

def get_keywords_top5(text, stopwords):

  min_count = 2     # 단어의 최소 출현 빈도수 (그래프 생성 시)
  max_length = 3    # 단어의 최대 길이
  beta = 0.99       # PageRank의 decaying factor beta
  max_iter = 15

  St = SentenceTokenizer(stopwords)
  korean_pattern = '[ㄱ-ㅎㅏ-ㅣ가-힣]+'
  is_hangul = bool(re.search(korean_pattern, text))
  
  if is_hangul is False:
    print("Error: Not written in Korean.")
    return
  
  sentences = St.text2sentences(text)
  nouns_sentence = St.get_nouns(sentences)

  # 명사 단위로 추출된 text에서 keywords 구분
  raw_keywords_dict = summarize_with_keywords(nouns_sentence,
                                              min_count=min_count,
                                              max_length=max_length,
                                              beta=beta,
                                              max_iter=max_iter,
                                              stopwords=stopwords,
                                              verbose=False)
  keywords_list = list(raw_keywords_dict.keys())
  top_5 = keywords_list[0:5]
  output = ''
  for keyword in top_5:
    output = output + keyword + ' '

  return output