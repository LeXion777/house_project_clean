# llama_model.py
import os
import time
import threading
import subprocess

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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login

# =========================================================
# 모델/LoRA 경로
# =========================================================
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# LoRA 파인튜닝 결과(기존과 동일)
LORA_PATH = os.path.join(os.path.dirname(__file__), "checkpoint-100")

# 파인튜닝을 “서버 부팅 시 자동 수행” 하고 싶다면,
# 학습 결과를 저장할 디렉토리를 별도로 두는 편이 관리가 좋음.
# (기존 checkpoint-100을 그대로 쓰려면 아래 OUTPUT_LORA_DIR을 LORA_PATH로 두면 됨)
OUTPUT_LORA_DIR = os.getenv("LORA_OUTPUT_DIR", LORA_PATH)

# “파인튜닝 완료” 마커 & 락 파일
FINETUNE_DONE_MARK = os.path.join(OUTPUT_LORA_DIR, ".finetune_done")
FINETUNE_LOCK_FILE = os.path.join(OUTPUT_LORA_DIR, ".finetune_lock")

tokenizer = None
model = None
_logged_in = False

# ✅ 동시 generate 보호(Transformers/GPU는 멀티스레드 동시 generate가 자주 문제남)
_generate_lock = threading.Lock()

# ✅ 모델 로딩 동시성 보호(부팅/요청이 겹쳐도 1번만 로드)
_load_lock = threading.Lock()

# ✅ 초기화 상태
_ready_event = threading.Event()
_init_error = None
_init_thread_started = False
_init_lock = threading.Lock()


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
# (옵션) 파인튜닝 트리거
# =========================================================
def _lora_checkpoint_exists(path: str) -> bool:
    # PeftModel.from_pretrained가 읽을 핵심 파일들 존재 여부로 체크
    if not os.path.isdir(path):
        return False
    needed = ["adapter_config.json"]
    return all(os.path.exists(os.path.join(path, f)) for f in needed)


def _acquire_file_lock(lock_path: str, timeout_sec: int = 3600) -> int:
    """
    매우 단순한 파일 락(프로세스간 중복 파인튜닝 방지).
    락 파일을 O_EXCL로 생성하면 한 프로세스만 성공.
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode())
            return fd
        except FileExistsError:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Finetune lock timeout: {lock_path}")
            time.sleep(2)


def _release_file_lock(fd: int, lock_path: str):
    try:
        os.close(fd)
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def maybe_finetune_lora():
    """
    ✅ 서버 부팅 시 “필요하면” LoRA 파인튜닝 1회 수행.
    - 이미 OUTPUT_LORA_DIR에 체크포인트가 있거나 .finetune_done이 있으면 스킵
    - 락으로 중복 실행 방지
    """
    # 기존 LoRA가 이미 있으면 스킵
    if os.path.exists(FINETUNE_DONE_MARK) or _lora_checkpoint_exists(OUTPUT_LORA_DIR):
        return

    os.makedirs(OUTPUT_LORA_DIR, exist_ok=True)

    lock_fd = _acquire_file_lock(FINETUNE_LOCK_FILE, timeout_sec=6 * 3600)
    try:
        # 락 잡고 다시 한 번 확인(경합 방지)
        if os.path.exists(FINETUNE_DONE_MARK) or _lora_checkpoint_exists(OUTPUT_LORA_DIR):
            return

        train_script = os.getenv("LORA_TRAIN_SCRIPT", "train_lora.py")
        train_data = os.getenv("LORA_TRAIN_DATA", "/workspace/data/train.jsonl")

        if not os.path.exists(train_script):
            raise RuntimeError(
                f"파인튜닝 스크립트({train_script})가 없습니다. "
                f"LORA_TRAIN_SCRIPT 환경변수로 경로를 지정하거나 스크립트를 추가하세요."
            )
        if not os.path.exists(train_data):
            raise RuntimeError(
                f"파인튜닝 데이터({train_data})가 없습니다. "
                f"LORA_TRAIN_DATA 환경변수로 경로를 지정하세요."
            )

        cmd = [
            "python", train_script,
            "--base_model", BASE_MODEL_NAME,
            "--train_data", train_data,
            "--output_dir", OUTPUT_LORA_DIR,
        ]

        print("[FINETUNE] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # 완료 마커
        with open(FINETUNE_DONE_MARK, "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    finally:
        _release_file_lock(lock_fd, FINETUNE_LOCK_FILE)


# =========================================================
# 모델 로딩 (Base + LoRA)
# =========================================================
def load_model():
    """
    ✅ Base + LoRA 로딩 (동시 호출에도 1회만 수행)
    """
    global tokenizer, model

    # 이미 로드된 경우(둘 다 있어야 함)
    if model is not None and tokenizer is not None:
        return

    with _load_lock:
        if model is not None and tokenizer is not None:
            return

        ensure_hf_login()

        print(f"[LLAMA_LOAD] base={BASE_MODEL_NAME}")
        print(f"[LLAMA_LOAD] lora_dir={OUTPUT_LORA_DIR}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # LoRA 어댑터 결합 (OUTPUT_LORA_DIR 사용)
        if not _lora_checkpoint_exists(OUTPUT_LORA_DIR):
            raise RuntimeError(f"LoRA 체크포인트가 없습니다: {OUTPUT_LORA_DIR}")

        model = PeftModel.from_pretrained(base_model, OUTPUT_LORA_DIR)
        model.eval()

        # ✅ LoRA 적용 확인 로그 (핵심)
        try:
            # peft 버전에 따라 속성명이 조금 다를 수 있어 try로 안전 처리
            adapters = getattr(model, "peft_config", None)
            if adapters:
                print(f"[LLAMA_LOAD] peft_adapters={list(adapters.keys())}")
            else:
                print("[LLAMA_LOAD] peft_adapters=UNKNOWN (peft_config not found)")
        except Exception as e:
            print(f"[LLAMA_LOAD] peft_adapters=ERROR {repr(e)}")

        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_LORA_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        print("[LLAMA_LOAD] ✅ model+tokenizer ready (base+LoRA applied)")


# =========================================================
# Prompt 구성 (Alpaca 학습 포맷과 호환)
# =========================================================
def build_prompt(chat_history, system_prompt):
    prompt = f"""### System:
{system_prompt}
"""
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"\n### Instruction:\n{content}\n"
        elif role == "assistant":
            prompt += f"\n### Response:\n{content}\n"
    prompt += "\n### Response:\n"
    return prompt


# =========================================================
# ✅ 서버 부팅 시 초기화 시작 (스레드)
# =========================================================
def startup(async_init: bool = True):
    """
    Flask가 올라올 때 한 번 호출해주면 됨.
    async_init=True면 서버는 바로 뜨고, 모델 준비는 백그라운드에서 진행.
    """
    global _init_thread_started

    with _init_lock:
        if _init_thread_started:
            return
        _init_thread_started = True

    def _init_job():
        global _init_error
        try:
            ensure_hf_login()
            maybe_finetune_lora()
            load_model()
            _ready_event.set()
        except Exception as e:
            _init_error = e
            print("[INIT_ERROR]", repr(e))

    if async_init:
        t = threading.Thread(target=_init_job, daemon=True)
        t.start()
    else:
        _init_job()


def is_ready() -> bool:
    return _ready_event.is_set()


def init_error():
    return _init_error


# =========================================================
# ✅ create_app()에서 바로 부를 수 있는 "완전 동기" 프리로드/워밍업 함수
# =========================================================
def warmup_model(
        do_warmup_generate: bool = True,
        max_new_tokens: int = 1,
):
    """
    ✅ 서버 부팅 시 미리 로드하고 ready를 올려두는 용도.
    - create_app()에서 warmup_model() 호출하면, views.py 수정 없이도 “첫 요청 지연”이 사라짐.
    - 내부적으로: HF 로그인 → (옵션) 필요 시 LoRA 파인튜닝 → 모델 로드 → ready set
    - do_warmup_generate=True면 아주 짧게 1토큰 생성해서 커널/캐시 워밍업까지 수행
    """
    global _init_error, _init_thread_started

    # 다른 곳에서 async startup을 이미 호출했더라도, 여기서 동기 보장 가능
    with _init_lock:
        _init_thread_started = True  # 중복 init 방지(의미상 "이미 init 경로 진입"으로 처리)

    try:
        ensure_hf_login()
        maybe_finetune_lora()
        load_model()

        if do_warmup_generate:
            # 아주 작은 generate로 워밍업(동시 generate 락 재사용)
            prompt = "### System:\nYou are a helpful assistant.\n\n### Instruction:\n안녕\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with _generate_lock, torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id
                )

        _ready_event.set()
        _init_error = None

    except Exception as e:
        _init_error = e
        # ready는 세트하지 않음(뷰에서 timeout/에러 처리 가능)
        print("[WARMUP_ERROR]", repr(e))
        raise


# =========================================================
# 추론 함수
# =========================================================
def generate_chat(
        chat_history,
        system_prompt,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=256,
        repetition_penalty=1.1,
        wait_ready: bool = True,
        wait_timeout_sec: int = 600,
):
    """
    wait_ready=True면 모델 준비될 때까지 최대 wait_timeout_sec 대기.
    False면 준비 안 됐을 때 즉시 예외 발생(views에서 503 처리하기 좋음)
    """
    if wait_ready:
        ok = _ready_event.wait(timeout=wait_timeout_sec)
        if not ok:
            raise RuntimeError("Model is not ready (timeout).")
    else:
        if not is_ready():
            raise RuntimeError("Model is not ready.")

    # 모델/토크나이저는 이미 load 되어 있어야 함
    prompt = build_prompt(chat_history, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 동시 generate 방지
    with _generate_lock, torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()
