import os
import json
import hashlib
import pkuseg
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# OpenAI API 키 설정
api_key = "your_api_key"

client = OpenAI(api_key=api_key)

# PKUSeg 초기화
seg = pkuseg.pkuseg(postag=True)  # 품사 태깅 활성화

# 시스템 프롬프트
SYSTEM_PROMPT = """你是中文句法分析专家。请严格遵循：
1. 保持原始词性标注不变
2. 依存分析必须基于给定词性序列
3. 依存分析的rel标签列表见附录
4. 树结构节点统一用：(PU 标点)

附录：
[rel标签体系]
AGT=施事 | PAT=受事 | EXP=主事 | CONT=客事
DATV=涉事 | LINK=系事 | TOOL=工具 | MATL=材料
MANN=方式 | TIME=时间 | LOC=空间 | MEAS=度量

[树结构示例]
例子1：(S(NP 他)(VP(PP(P 朝)(NP(NP 书包)(F 里)))(VP(VP(V 塞)(V 进)) (NP (AP (D 很) (A 多))(NP 书))))(W 。))
例子2：(S (NP 张三)(VP (V 是)(NP (CS (NP 县长)(VP (V 派)(V 来)))(ude 的))))

[依存关系示例]
例子1：(S(NP[dep:#塞#, rel:施事] 他)(VP(PP[dep:#塞#, rel:处所] (P 朝)(NP(NP 书包)(F 里)))(VP(VP(V[dep:#塞#, rel:核心] 塞)(V[dep:#塞#, rel:趋向] 进)) (NP [dep:#进#, rel:主事] (AP (D 很) (A 多))(NP 书))))(W 。))
例子2：(S (NP[[dep:#派#, rel:受事],[dep:#来#, rel:主事],[dep:#是#, rel:主事]] 张三)(VP (V 是)(NP[dep:#是#, rel:系事] (CS (NP[dep:#派#, rel:施事] 县长)(VP (V 派)(V 来)))(ude 的))))
"""

# 캐시 시스템
class AnalysisCache:
    def __init__(self, cache_file="analysis_cache.json"):
        self.cache_file = cache_file
        self.cache = {}  # 항상 빈 캐시로 시작

    def get_key(self, sentence, pos_result):
        combined = f"{sentence}||{pos_result}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def save(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

cache = AnalysisCache()

def pkuseg_pos_tagging(sentence):
    """pkuseg를 이용한 품사 태깅"""
    tagged = seg.cut(sentence)
    pos_seq = ",".join([f"({idx+1},{word}/{pos})" for idx, (word, pos) in enumerate(tagged)])
    return pos_seq

def gpt_analysis(sentence, pos_result):
    """GPT를 이용한 심층 분석"""
    cache_key = cache.get_key(sentence, pos_result)

    if cache_key in cache.cache:
        return cache.cache[cache_key]

    user_prompt = f"""请分析如下的句子:

    sentence: {sentence}
    pos_tagging: {pos_result}

    输出格式:
    {{
      "parse_tree": "(S(...))",
      "dependency_tree": "((S(NP[dep:#...#, rel:role] A)"
    }}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=500
        )

        # JSON 응답 처리
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            print("JSON 디코딩 오류 발생. 응답 내용:", response.choices[0].message.content)
            result = {"parse_tree": "", "dependency_tree": ""}

        cache.cache[cache_key] = result
        return result
    except Exception as e:
        print(f"API 오류 발생: {str(e)}")
        return {"parse_tree": "", "dependency_tree": ""}

def process_sentence(sentence, index):
    """단일 문장 처리 파이프라인"""
    pos_tags = pkuseg_pos_tagging(sentence)
    analysis = gpt_analysis(sentence, pos_tags)

    return {
        "data_id": f"DATA_{index+1:03d}",
        "data_text": sentence,
        "annotations": [
            {"tags": []},
            {"labels": []},
            {"relations": []}
        ],
        "syntactic_annotations": [
            {"pos_tagging": pos_tags},
            {"parse_tree": analysis.get("parse_tree", "")},
            {"dependency_tree": analysis.get("dependency_tree", "")}
        ]
    }

def main(input_file, output_file):
    """메인 처리 함수"""
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
        sentences = df['text'].tolist()
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

    results = []
    for idx, sent in enumerate(tqdm(sentences, desc="문장 처리 진행 중")):
        try:
            result = process_sentence(sent, idx)
            print(result)
            results.append(result)
        except Exception as e:
            print(f"문장 {idx+1} 처리 실패: {str(e)}")
            results.append({
                "data_id": f"DATA_{idx+1:03d}",
                "data_text": sent,
                "annotations": [
                    {"tags": []},
                    {"labels": []},
                    {"relations": []}
                ],
                "syntactic_annotations": [
                    {"pos_tagging": ""},
                    {"parse_tree": ""},
                    {"dependency_tree": ""}
                ]
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    cache.save()

if __name__ == "__main__":
    main("input.txt", "output.json")
