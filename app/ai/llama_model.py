import os

# =========================================================
# RunPod 필수: Hugging Face 캐시 / TMP 경로 강제 고정
# =========================================================
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
os.environ["TMPDIR"] = "/workspace/tmp"

os.makedirs("/workspace/hf_cache", exist_ok=True)
os.makedirs("/workspace/tmp", exist_ok=True)

# =========================================================
# Imports
# =========================================================
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from huggingface_hub import login

# =========================================================
# 모델 경로 설정
# =========================================================
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# LoRA 파인튜닝 결과
LORA_PATH = os.path.join(
    os.path.dirname(__file__),
    "checkpoint-50"
)

tokenizer = None
model = None
_logged_in = False


# =========================================================
# Hugging Face 인증
# =========================================================
def ensure_hf_login():
    global _logged_in
    if _logged_in:
        return

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN 환경변수가 설정되어 있지 않습니다.")

    login(token=hf_token)
    _logged_in = True


# =========================================================
# 모델 로딩 (Base + LoRA)
# =========================================================
def load_model():
    global tokenizer, model

    if model is not None:
        return

    ensure_hf_login()

    # 4bit Quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Base LLaMA 3 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # LoRA 어댑터 결합
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH
    )

    model.eval()

    # Tokenizer (LoRA 경로 기준)
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id


# =========================================================
# Prompt 구성 (선생님 Alpaca 학습 포맷과 호환)
# =========================================================
def build_prompt(chat_history):
    """
    Instruction Fine-Tuned 모델용 Prompt
    """
    last_user_msg = chat_history[-1]["content"]

    prompt = f"""Below is an instruction that describes a task.

### Instruction:
{last_user_msg}

### Response:
"""
    return prompt


# =========================================================
# 추론 함수 (하이퍼파라미터 실험 핵심)
# =========================================================
def generate_chat(
        chat_history,
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        repetition_penalty=1.1
):
    load_model()

    prompt = build_prompt(chat_history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded.split("### Response:")[-1].strip()
