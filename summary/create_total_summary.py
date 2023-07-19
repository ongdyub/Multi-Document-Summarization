# Set-up create_total_summary

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BartForConditionalGeneration
import numpy as np
import torch

###############################################################################################################################
### 최종 요약문 생성 과정 중 1 / 2 과정 ###
### create_total_summary(summary_list) 함수가 동작 ###
### Input : 개별 요약문 list ex) ['요약문1','요약문2',...,'요약문5']
### Output : '최종 요약문 string'
###############################################################################################################################
###############################################################################################################################

# 두 문장 사이의 cos 유사도를 계산하는 함수
# Input : 두 문장의 임베딩 벡터
# Output : cos값
def cos_sim(s_1, s_2):
    vector1 = np.array(s_1)
    vector2 = np.array(s_2)

    # NumPy 배열을 리스트로 변환
    vector1_list = vector1.tolist()
    vector2_list = vector2.tolist()

    # 벡터의 크기 계산
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # 코사인 유사도 계산
    cosine_similarity = np.dot(vector1_list, vector2_list) / (magnitude1 * magnitude2)
    
    return cosine_similarity

# 유사도가 높은 list index를 만드는 함수
# 최종 요약문의 과정 2. Grouping 의 index 생성
# Input : target list 쌍, base_list 는 이미 만들어진 집합 ex) [3,5] , [[1,2][3,4]]
# Output : [[1,2],[3,4,5]]
def merge_list(target_list, base_list):
    if(len(base_list) == 0):
        base_list.append(target_list)
        return base_list

    idx_1 = 0
    idx_2 = 0
    cnt = 0

    for idx, base in enumerate(base_list):
        for target in target_list:
            if(cnt == 0 and target in base):
                idx_1 = idx
                cnt = 1
                continue
            if(cnt == 1 and target in base):
                idx_2 = idx
                cnt = 2
                continue
        if(cnt == 2):
            break
    
    if(cnt == 0):
        base_list.append(target_list)
        return base_list
    elif(cnt == 1):
        base_list[idx_1] = list(set(target_list) | set(base_list[idx_1]))
        return base_list
    elif(cnt == 2 and idx_1 == idx_2):
        return base_list
    else:
        base_list[idx_1] = list(set(base_list[idx_1]) | set(base_list[idx_2]))
        base_list.pop(idx_2)
        return base_list


# 그룹화 리스트를 반환하는 함수
# 최종 요약문의 과정 2. Grouping 문장의 list를 반환
# Input : 개별 요약문 list ex) ['요약문1','요약문2',...,'요약문5']
# Output : grouping 된 요약문 list의 list ex) [['요약문1', '요약문3'],['요약문2'],['요약문4','요약문5']]
def make_related_list(summary_list, model):
    
    embedding_list = model.encode(summary_list)

    related_list = []

    for idx_1, embedding_1 in enumerate(embedding_list):
        for idx_2, embedding_2 in enumerate(embedding_list):
            if(idx_1 >= idx_2):
                continue

            similarity = cos_sim(embedding_1, embedding_2)

            if(similarity > 0.6):
                related_list = merge_list([idx_1,idx_2],related_list)
    
    if(len(related_list) == 0):
        output_list = []
        for summary in summary_list:
            temp_list =[]
            temp_list.append(summary)
            output_list.append(temp_list)
        return output_list

    final_idx_list = list(range(len(summary_list)))

    for related_idx_list in related_list:
        for idx in related_idx_list:
            final_idx_list.remove(idx)
    
    for final_idx in final_idx_list:
        temp = []
        temp.append(final_idx)
        related_list.append(temp)
    
    output_summary_group_list = []
    for relate in related_list:
        summary_group = []
        for idx in relate:
            summary_group.append(summary_list[idx])
        output_summary_group_list.append(summary_group)

    return output_summary_group_list

# 최종 요약문을 만드는 함수
# 최종 요약문의 과정 중 3 재요약과 4 최종 요약문 생성
# Input : grouping 된 요약문 list의 list ex) [['요약문1', '요약문3'],['요약문2'],['요약문4','요약문5']]
# Output : 최종 요약문 'string'
def make_total_summary(group_list, summary_model, tokenizer):
    model = summary_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = ''

    for sentence_list in group_list:
        if(len(sentence_list) <= 1):
            output += sentence_list[0]
            continue

        input_text = ''
        for sentence in sentence_list:
            input_text += sentence
        raw_input_ids = tokenizer.encode(input_text)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

        summary_ids = model.generate(torch.tensor([input_ids]).to(device),
                                max_length=1024,
                                early_stopping=True,
                                repetition_penalty=2.0)
        summ = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        
        if(summ.find("다.") != -1):
            summ = summ.split("다.")[0] + "다."

        output += summ

    output = output.replace('다.', '다. ')
    return output

# 최종 단계의 함수
# Input : 개별 요약문 list ex) ['요약문1','요약문2',...,'요약문5']
# Output : '최종 요약문 string'
def create_total_summary(summary_list, bert_model, summary_model, tokenizer):

    group_list = make_related_list(summary_list, bert_model)
    total_summary = make_total_summary(group_list, summary_model, tokenizer)
    return total_summary
