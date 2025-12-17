import torch
from transformers import pipeline
import os

print(">>> AI 모델 로딩 시작...")

DEVICE = 0 if torch.cuda.is_available() else -1

# 1) QA
qa_pipeline = pipeline(
    "question-answering",
    model="monologg/koelectra-base-v3-finetuned-korquad",
    tokenizer="monologg/koelectra-base-v3-finetuned-korquad",
    device=DEVICE
)

# 2) 텍스트 생성
textgen_pipeline = pipeline(
    "text-generation",
    model="skt/kogpt2-base-v2",
    tokenizer="skt/kogpt2-base-v2",
    device=DEVICE
)

# 3) 번역
translate_pipeline = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    tokenizer="facebook/nllb-200-distilled-600M",
    src_lang="kor_Hang",
    tgt_lang="eng_Latn",
    device=DEVICE
)

# 4) 감정 분석
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="beomi/KcELECTRA-base-v2022",
    tokenizer="beomi/KcELECTRA-base-v2022",
    device=DEVICE
)

# 5) NER
ner_pipeline = pipeline(
    "ner",
    model="Davlan/bert-base-multilingual-cased-ner-hrl",
    tokenizer="Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple",
    device=DEVICE
)

print(">>> AI 모델 로딩 완료!")
