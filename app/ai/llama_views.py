# llama_views.py
import logging
from flask import Blueprint, render_template, request, session, jsonify, redirect, url_for
from .llama_model import generate_chat

bp = Blueprint("llama", __name__)

# ✅ 터미널 로그 핸들러
log = logging.getLogger("llama")
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
log.setLevel(logging.INFO)

# ✅ 디폴트 하이퍼파라미터 (generate_chat 시그니처에 맞춘 5개)
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
    "repetition_penalty": 1.1,
}

# ✅ 디폴트 System Prompt
DEFAULT_SYSTEM_PROMPT = ("You are a helpful assistant. Always reply in Korean unless the user explicitly asks you to use another language.")

# ✅ 입력 검증/캐스팅용 스펙(서버 안전장치)
_PARAM_SPECS = {
    "temperature": {"type": float, "min": 0.0, "max": 2.0},
    "top_p": {"type": float, "min": 0.0, "max": 1.0},
    "top_k": {"type": int, "min": 0, "max": 200},
    "max_tokens": {"type": int, "min": 1, "max": 4096},
    "repetition_penalty": {"type": float, "min": 0.5, "max": 2.5},
}


def _clamp(v, vmin, vmax):
    return vmax if v > vmax else vmin if v < vmin else v


def _coerce_param(data: dict, key: str):
    default = DEFAULT_PARAMS[key]
    spec = _PARAM_SPECS[key]
    raw = data.get(key, default)

    try:
        if spec["type"] is int:
            val = int(float(raw))  # "50.0" 같은 문자열도 안전 처리
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


@bp.route("/llama", methods=["GET", "POST"])
def llama_chat():
    # ✅ 세션에는 "대화내역만"
    session.setdefault("chat_history", [])

    # ✅ JSON 요청: 채팅 Send
    if request.method == "POST" and request.is_json:
        data = request.get_json(silent=True) or {}

        # (UI 호환) JSON reset_chat도 처리
        if data.get("action") == "reset_chat":
            session["chat_history"] = []
            session.modified = True
            log.info("[RESET_CHAT:JSON] chat_history cleared")
            return jsonify({"ok": True})

        user_input = (data.get("prompt") or "").strip()
        if not user_input:
            return jsonify({"answer": ""})

        system_prompt, params = _extract_request_config(data)

        sp_preview = (system_prompt[:180] + "…") if len(system_prompt) > 180 else system_prompt
        log.info("[SEND] system_prompt=%r", sp_preview)
        log.info("[SEND] params=%s", params)

        session["chat_history"].append({"role": "user", "content": user_input})

        assistant_reply = generate_chat(
            session["chat_history"],
            system_prompt=system_prompt,
            **params
        )

        session["chat_history"].append({"role": "assistant", "content": assistant_reply})
        session.modified = True
        return jsonify({"answer": assistant_reply})

    # ✅ FORM 요청: Reset Chat만 지원
    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "reset_chat":
            session["chat_history"] = []
            session.modified = True
            log.info("[RESET_CHAT:FORM] chat_history cleared")
            return redirect(url_for("llama.llama_chat"))

        return redirect(url_for("llama.llama_chat"))

    # ✅ GET: 항상 디폴트로 렌더링
    return render_template(
        "llama/llama.html",
        chat_history=session.get("chat_history", []),

        # ✅ 새 템플릿 변수명(이번에 사용하는 이름)
        default_params=DEFAULT_PARAMS,
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,

        # ✅ (혹시 템플릿이 예전 걸로 남아있어도 깨지지 않게) 구 변수명도 같이 전달
        params=DEFAULT_PARAMS,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
