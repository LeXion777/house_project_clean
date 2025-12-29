# llama_model.py
import os
import time
import threading
import subprocess
from typing import Any, Dict, List

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
# Context logging controls
# =========================================================
# 0: off, 1: compact(default), 2: verbose
_CTX_LOG_LEVEL = int(os.getenv("LLAMA_CTX_LOG", "1"))
# how many recent messages to show in terminal
_CTX_LOG_LAST_N = int(os.getenv("LLAMA_CTX_LOG_LAST_N", "6"))
# preview chars for each message line
_CTX_LOG_MSG_PREVIEW = int(os.getenv("LLAMA_CTX_LOG_MSG_PREVIEW", "180"))
# preview chars for prompt head/tail
_CTX_LOG_PROMPT_PREVIEW = int(os.getenv("LLAMA_CTX_LOG_PROMPT_PREVIEW", "700"))
# if set, print full prompt (NOT recommended)
_CTX_LOG_FULL = os.getenv("LLAMA_CTX_LOG_FULL", "0").strip() in ("1", "true", "True", "yes", "Y")

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

# 서버 최초 부팅 기본 체크포인트(요구사항: 훼손 금지)
DEFAULT_FINETUNE_ID = "ckpt_100"

# llama_model.py와 같은 폴더에 checkpoint-100/50이 있다고 가정
_THIS_DIR = os.path.dirname(__file__)
_DEFAULT_CKPT100_DIR = os.getenv("LORA_OUTPUT_DIR", os.path.join(_THIS_DIR, "checkpoint-100"))


def _derive_ckpt50_dir(ckpt100_dir: str) -> str:
    """
    checkpoint-50은 checkpoint-100과 동일 경로에서 폴더명만 다르다고 가정.
    - env로 ckpt100 경로를 바꿔도, 그 부모에서 checkpoint-50을 찾도록 함.
    """
    try:
        base = os.path.basename(os.path.normpath(ckpt100_dir))
        parent = os.path.dirname(os.path.normpath(ckpt100_dir))
        if base == "checkpoint-100":
            return os.path.join(parent, "checkpoint-50")
    except Exception:
        pass
    return os.path.join(_THIS_DIR, "checkpoint-50")


_FINETUNE_DIR_MAP = {
    "ckpt_100": _DEFAULT_CKPT100_DIR,
    "ckpt_50": _derive_ckpt50_dir(_DEFAULT_CKPT100_DIR),
}

# 기존 구조 유지) "부팅 시 필요하면 학습"은 ckpt_100(OUTPUT_LORA_DIR)만 대상으로 둠
OUTPUT_LORA_DIR = _FINETUNE_DIR_MAP["ckpt_100"]
FINETUNE_DONE_MARK = os.path.join(OUTPUT_LORA_DIR, ".finetune_done")
FINETUNE_LOCK_FILE = os.path.join(OUTPUT_LORA_DIR, ".finetune_lock")

# =========================================================
# Globals
# =========================================================
tokenizer = None  # base tokenizer (권장)
base_model = None  # base quantized model
model = None  # PeftModel (base + selected adapter)
_current_finetune_id = None

_logged_in = False

# 동시 generate 보호(Transformers/GPU는 멀티스레드 동시 generate가 자주 문제남)
_generate_lock = threading.Lock()

# 모델 로딩/스위칭 동시성 보호
_load_lock = threading.Lock()

# 초기화 상태
_ready_event = threading.Event()
_init_error = None
_init_thread_started = False
_init_lock = threading.Lock()


# =========================================================
# Small helpers for context logging
# =========================================================
def _one_line(s: str) -> str:
    return " ".join((s or "").replace("\r", "\n").split())


def _preview(s: str, n: int) -> str:
    s = _one_line(s)
    return s if len(s) <= n else (s[:n] + "…")


def _log_context(chat_history: List[Dict[str, str]], system_prompt: str, prompt: str, input_len: int):
    """
    터미널에 '이번 요청에서 사용된 컨텍스트'를 사람이 보기 쉽게 출력.
    """
    if _CTX_LOG_LEVEL <= 0:
        return

    try:
        total_msgs = len(chat_history or [])
        sys_prev = _preview(system_prompt or "", 220)

        print(f"[LLAMA_CTX] system_prompt='{sys_prev}'")
        print(f"[LLAMA_CTX] history_messages={total_msgs}, input_tokens={input_len}")

        # last N messages preview
        last_n = max(0, _CTX_LOG_LAST_N)
        if last_n > 0 and total_msgs > 0:
            slice_msgs = (chat_history or [])[-last_n:]
            print(f"[LLAMA_CTX] last_{len(slice_msgs)}_messages:")
            for i, m in enumerate(slice_msgs, 1):
                role = m.get("role", "?")
                content = m.get("content", "")
                print(f"  - {i:02d}. {role}: {_preview(str(content), _CTX_LOG_MSG_PREVIEW)}")

        # prompt preview
        if _CTX_LOG_FULL:
            print("[LLAMA_CTX] prompt_full_begin")
            print(prompt)
            print("[LLAMA_CTX] prompt_full_end")
        else:
            prev_len = _CTX_LOG_PROMPT_PREVIEW if _CTX_LOG_LEVEL >= 2 else min(_CTX_LOG_PROMPT_PREVIEW, 450)
            head = prompt[:prev_len]
            tail = prompt[-prev_len:] if len(prompt) > prev_len else ""
            print("[LLAMA_CTX] prompt_head_preview:")
            print(head)
            if tail:
                print("[LLAMA_CTX] prompt_tail_preview:")
                print(tail)

    except Exception as e:
        print(f"[LLAMA_CTX] (log failed) {repr(e)}")


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
# (옵션) 파인튜닝 트리거 (ckpt_100만)
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
    서버 부팅 시 “필요하면” LoRA 파인튜닝 1회 수행 (ckpt_100만)
    - 이미 OUTPUT_LORA_DIR에 체크포인트가 있거나 .finetune_done이 있으면 스킵
    - 락으로 중복 실행 방지
    """
    if os.path.exists(FINETUNE_DONE_MARK) or _lora_checkpoint_exists(OUTPUT_LORA_DIR):
        return

    os.makedirs(OUTPUT_LORA_DIR, exist_ok=True)

    lock_fd = _acquire_file_lock(FINETUNE_LOCK_FILE, timeout_sec=6 * 3600)
    try:
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

        with open(FINETUNE_DONE_MARK, "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    finally:
        _release_file_lock(lock_fd, FINETUNE_LOCK_FILE)


# =========================================================
# Finetune ID -> dir
# =========================================================
def _resolve_lora_dir(finetune_id: str) -> str:
    if not isinstance(finetune_id, str):
        finetune_id = DEFAULT_FINETUNE_ID
    finetune_id = finetune_id.strip() or DEFAULT_FINETUNE_ID
    return _FINETUNE_DIR_MAP.get(finetune_id, _FINETUNE_DIR_MAP[DEFAULT_FINETUNE_ID])


# =========================================================
# Base 모델 로딩 (1회)
# =========================================================
def _load_base_and_tokenizer_once():
    global base_model, tokenizer

    if base_model is not None and tokenizer is not None:
        return

    ensure_hf_login()

    print(f"[LLAMA_LOAD] base={BASE_MODEL_NAME}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # tokenizer는 base에서 로드 (LoRA 폴더에 tokenizer가 없을 수 있으니 안정적)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # pad 토큰 보강
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base_model.config.pad_token_id = tokenizer.eos_token_id
    except Exception:
        pass


# =========================================================
# LoRA 어댑터 로딩/스위칭
# =========================================================
def load_model(finetune_id: str = DEFAULT_FINETUNE_ID):
    """
    Base(1회) + 선택 LoRA(요청마다 스위칭 가능) 로딩
    - finetune_id가 바뀌면 기존 PeftModel을 폐기하고 새 adapter로 교체
    """
    global model, _current_finetune_id

    with _load_lock:
        _load_base_and_tokenizer_once()

        target_id = (finetune_id or DEFAULT_FINETUNE_ID).strip()
        target_dir = _resolve_lora_dir(target_id)

        # 이미 같은 finetune이면 스킵
        if model is not None and _current_finetune_id == target_id:
            return

        if not _lora_checkpoint_exists(target_dir):
            raise RuntimeError(f"LoRA 체크포인트가 없습니다: {target_dir}")

        # 기존 모델 정리(“종료” 의미: adapter 언로드 & 메모리 반환)
        if model is not None:
            try:
                del model
            except Exception:
                pass
            model = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        print(f"[LLAMA_LOAD] lora_dir={target_dir} (finetune_id={target_id})")

        # 새 LoRA 어댑터 결합
        model = PeftModel.from_pretrained(base_model, target_dir)
        model.eval()
        _current_finetune_id = target_id

        # LoRA 적용 확인 로그
        try:
            adapters = getattr(model, "peft_config", None)
            if adapters:
                print(f"[LLAMA_LOAD] peft_adapters={list(adapters.keys())}")
            else:
                print("[LLAMA_LOAD] peft_adapters=UNKNOWN (peft_config not found)")
        except Exception as e:
            print(f"[LLAMA_LOAD] peft_adapters=ERROR {repr(e)}")

        print("[LLAMA_LOAD] model ready (base+LoRA applied)")


def ensure_finetune_loaded(finetune_id: str):
    """
    요청 finetune_id가 현재 로드된 것과 다르면 스위칭.
    """
    target_id = (finetune_id or DEFAULT_FINETUNE_ID).strip() or DEFAULT_FINETUNE_ID
    if model is not None and _current_finetune_id == target_id:
        return
    load_model(target_id)


# =========================================================
# Prompt 구성 (Alpaca 학습 포맷과 호환)
# =========================================================
def build_prompt(chat_history: List[Dict[str, str]], system_prompt: str) -> str:
    """
    chat_history: [{"role":"user"|"assistant","content":"..."}] (여러 턴)
    Alpaca 스타일(Instruction/Response)을 여러 번 반복하는 형태로 구성.
    """
    sp = (system_prompt or "").strip()
    prompt = f"""### System:
{sp}
"""

    for msg in (chat_history or []):
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content:
            continue

        if role == "user":
            prompt += f"\n### Instruction:\n{content}\n"
        elif role == "assistant":
            prompt += f"\n### Response:\n{content}\n"

    # 마지막에 모델이 이어서 답하도록 Response 헤더로 끝냄
    prompt += "\n### Response:\n"
    return prompt


def _postprocess_alpaca_reply(text: str) -> str:
    """
    모델이 다음 턴(### Instruction / ### System 등)까지 생성해버릴 때 컷.
    """
    if not text:
        return ""

    stop_markers = [
        "\n### Instruction:",
        "\n### System:",
        "\n### Response:",
    ]
    cut = None
    for m in stop_markers:
        idx = text.find(m)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)

    if cut is not None:
        text = text[:cut]

    return text.strip()


# =========================================================
# 서버 부팅 시 초기화 시작 (스레드)
# =========================================================
def startup(async_init: bool = True):
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
            load_model(DEFAULT_FINETUNE_ID)
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
# 동기 워밍업
# =========================================================
def warmup_model(
        do_warmup_generate: bool = True,
        max_new_tokens: int = 1,
):
    global _init_error, _init_thread_started

    with _init_lock:
        _init_thread_started = True

    try:
        ensure_hf_login()
        maybe_finetune_lora()
        load_model(DEFAULT_FINETUNE_ID)

        if do_warmup_generate:
            prompt = "### System:\nYou are a helpful assistant.\n\n### Instruction:\n안녕\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with _generate_lock, torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

        _ready_event.set()
        _init_error = None

    except Exception as e:
        _init_error = e
        print("[WARMUP_ERROR]", repr(e))
        raise


# =========================================================
# 추론 함수
# =========================================================
def generate_chat(
        chat_history,
        system_prompt,
        finetune_id: str = DEFAULT_FINETUNE_ID,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=256,
        repetition_penalty=1.1,
        wait_ready: bool = True,
        wait_timeout_sec: int = 600,
):
    if wait_ready:
        ok = _ready_event.wait(timeout=wait_timeout_sec)
        if not ok:
            raise RuntimeError("Model is not ready (timeout).")
    else:
        if not is_ready():
            raise RuntimeError("Model is not ready.")

    with _generate_lock:
        requested_id = (finetune_id or DEFAULT_FINETUNE_ID).strip() or DEFAULT_FINETUNE_ID
        before_id = _current_finetune_id

        ensure_finetune_loaded(requested_id)
        after_id = _current_finetune_id

        if before_id != after_id:
            print(f"[LLAMA_GEN] finetune switched: {before_id} -> {after_id} (requested={requested_id})")
        else:
            print(f"[LLAMA_GEN] finetune used: {after_id} (requested={requested_id})")

        prompt = build_prompt(chat_history, system_prompt)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        # 컨텍스트 로그 출력
        _log_context(chat_history=chat_history, system_prompt=system_prompt, prompt=prompt, input_len=input_len)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = outputs[0][input_len:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return _postprocess_alpaca_reply(reply)
