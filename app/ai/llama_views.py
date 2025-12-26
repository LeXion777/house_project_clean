# llama_views.py
import logging
from typing import Any, Dict, List

from flask import Blueprint, render_template, request, session, jsonify, redirect, url_for
from .llama_model import generate_chat  # generate_chat이 finetune_id 지원

bp = Blueprint("llama", __name__)

log = logging.getLogger("llama")
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
log.setLevel(logging.INFO)

DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
    "repetition_penalty": 1.1,
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Always reply in Korean unless the user explicitly asks you to use another language."
)

# 디폴트 Fine-tuning 체크포인트 (부팅 기본값: ckpt_100 유지)
DEFAULT_FINETUNE_ID = "ckpt_100"

FINETUNE_PRESETS = [
    {
        "id": "ckpt_50",
        "title": "Checkpoint 50",
        "badge": "step 50/100",
        "recommended": False,
        "short": "동일 Instruction SFT(run) 중간 저장본 (global_step=50)",
        "detail_kv": [
            {"k": "Snapshot", "v": "global_step=50 (max_steps=100, save_steps=50)"},
            {"k": "Training", "v": "Instruction SFT (Alpaca template)"},
            {"k": "Base", "v": "Meta-Llama-3-8B (4-bit NF4)"},
            {"k": "PEFT", "v": "LoRA (PEFT) adapter-only (r=16, alpha=16)"},
            {"k": "Objective", "v": 'completion-only loss on Response span (after "### Response")'},
        ],
    },
    {
        "id": "ckpt_100",
        "title": "Checkpoint 100",
        "badge": "step 100/100",
        "recommended": True,
        "short": "동일 Instruction SFT(run) 최종 저장본 (global_step=100, training end)",
        "detail_kv": [
            {"k": "Snapshot", "v": "global_step=100 (training end)"},
            {"k": "Diff", "v": "ckpt_50과 학습 설정 동일, 저장 step만 다름 (50 vs 100)"},
            {"k": "Training", "v": "Instruction SFT (Alpaca template)"},
            {"k": "Base", "v": "Meta-Llama-3-8B (4-bit NF4)"},
            {"k": "PEFT", "v": "LoRA (PEFT) adapter-only (r=16, alpha=16)"},
            {"k": "Objective", "v": 'completion-only loss on Response span (after "### Response")'},
        ],
    },
]

_PARAM_SPECS = {
    "temperature": {"type": float, "min": 0.0, "max": 2.0},
    "top_p": {"type": float, "min": 0.0, "max": 1.0},
    "top_k": {"type": int, "min": 0, "max": 200},
    "max_tokens": {"type": int, "min": 1, "max": 4096},
    "repetition_penalty": {"type": float, "min": 0.5, "max": 2.5},
}

# 컨텍스트 방어적 제한 (프론트가 쿠키로 보낸 history 기준)
MAX_CONTEXT_MESSAGES = 24  # user/assistant 합쳐서 최근 24개(=약 12턴)
MAX_MESSAGE_CHARS = 2000  # 메시지 1개 최대 길이
MAX_SESSION_MESSAGES = 120  # 세션에는 조금 더 넉넉히 보관(뷰 렌더링용)


def _clamp(v, vmin, vmax):
    return vmax if v > vmax else vmin if v < vmin else v


def _coerce_param(data: dict, key: str):
    default = DEFAULT_PARAMS[key]
    spec = _PARAM_SPECS[key]
    raw = data.get(key, default)

    try:
        if spec["type"] is int:
            val = int(float(raw))
        else:
            val = float(raw)
        val = _clamp(val, spec["min"], spec["max"])
        return val
    except Exception:
        return default


def _extract_request_config(data: dict):
    sp = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    if not isinstance(sp, str):
        sp = DEFAULT_SYSTEM_PROMPT
    sp = sp.strip() or DEFAULT_SYSTEM_PROMPT

    params = {k: _coerce_param(data, k) for k in DEFAULT_PARAMS.keys()}
    return sp, params


def _sanitize_finetune_id(raw_id: str) -> str:
    if not isinstance(raw_id, str):
        return DEFAULT_FINETUNE_ID
    raw_id = raw_id.strip()
    allowed = {p["id"] for p in FINETUNE_PRESETS}
    return raw_id if raw_id in allowed else DEFAULT_FINETUNE_ID


def _sanitize_history(raw: Any,
                      max_messages: int = MAX_CONTEXT_MESSAGES,
                      max_chars_per_msg: int = MAX_MESSAGE_CHARS) -> List[Dict[str, str]]:
    """
    history: [{role:"user"|"assistant", content:"..."}] 형태만 허용
    """
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg]
        out.append({"role": role, "content": content})

    if len(out) > max_messages:
        out = out[-max_messages:]
    return out


def _ensure_last_user_message(history: List[Dict[str, str]], user_input: str) -> List[Dict[str, str]]:
    """
    프론트가 이미 user_input을 history에 포함해서 보낼 수 있으니(쿠키 push 후 전송),
    마지막이 동일 user_input이면 중복 append를 막는다.
    """
    if not user_input:
        return history

    if history:
        last = history[-1]
        if last.get("role") == "user" and last.get("content") == user_input:
            return history

    history.append({"role": "user", "content": user_input})
    # safety: 컨텍스트 길이 유지
    if len(history) > MAX_CONTEXT_MESSAGES:
        history = history[-MAX_CONTEXT_MESSAGES:]
    return history


@bp.route("/llama", methods=["GET", "POST"])
def llama_chat():
    session.setdefault("chat_history", [])

    # JSON 요청(=fetch) 처리
    if request.method == "POST" and request.is_json:
        data = request.get_json(silent=True) or {}

        # JSON reset(현재 프론트는 FORM reset을 쓰지만 혹시 대비)
        if data.get("action") == "reset_chat":
            session["chat_history"] = []
            session.modified = True
            log.info("[RESET_CHAT:JSON] chat_history cleared")
            return jsonify({"ok": True})

        user_input = (data.get("prompt") or "").strip()
        if not user_input:
            return jsonify({"answer": ""})

        system_prompt, params = _extract_request_config(data)
        finetune_id = _sanitize_finetune_id(data.get("finetune_id", DEFAULT_FINETUNE_ID))

        # 핵심: 프론트가 보낸 history가 있으면 그걸 우선 사용(쿠키 컨텍스트)
        client_history = _sanitize_history(data.get("history"))
        if client_history:
            working_history = client_history
            log.info("[HISTORY] using client history (len=%d)", len(working_history))
        else:
            # fallback: 기존 세션 히스토리 사용
            working_history = _sanitize_history(session.get("chat_history", []), max_messages=MAX_SESSION_MESSAGES)
            log.info("[HISTORY] using session history (len=%d)", len(working_history))

        # 이번 입력은 항상 마지막 user로 보장(중복이면 추가 안 함)
        working_history = _ensure_last_user_message(working_history, user_input)

        log.info("[SEND] finetune_id=%s", finetune_id)
        sp_preview = (system_prompt[:180] + "…") if len(system_prompt) > 180 else system_prompt
        log.info("[SEND] system_prompt=%r", sp_preview)
        log.info("[SEND] params=%s", params)

        # 모델 호출 (history + system_prompt + finetune_id)
        assistant_reply = generate_chat(
            working_history,
            system_prompt=system_prompt,
            finetune_id=finetune_id,
            **params,
        )

        # 답변 append
        if not isinstance(assistant_reply, str):
            assistant_reply = str(assistant_reply)
        assistant_reply = assistant_reply.strip()

        working_history.append({"role": "assistant", "content": assistant_reply})
        if len(working_history) > MAX_SESSION_MESSAGES:
            working_history = working_history[-MAX_SESSION_MESSAGES:]

        # 세션에도 미러링(페이지 새로고침 시 chat_history 렌더링용)
        session["chat_history"] = working_history
        session.modified = True

        return jsonify({"answer": assistant_reply})

    # FORM POST 처리(Reset Chat 버튼)
    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "reset_chat":
            session["chat_history"] = []
            session.modified = True
            log.info("[RESET_CHAT:FORM] chat_history cleared")
            return redirect(url_for("llama.llama_chat"))

        return redirect(url_for("llama.llama_chat"))

    # GET: 페이지 렌더링
    return render_template(
        "llama/llama.html",
        chat_history=session.get("chat_history", []),

        finetune_presets=FINETUNE_PRESETS,
        default_finetune_id=DEFAULT_FINETUNE_ID,

        default_params=DEFAULT_PARAMS,
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,

        params=DEFAULT_PARAMS,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
