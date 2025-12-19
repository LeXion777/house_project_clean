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

# ✅ 디폴트 System Prompt (요청한 문구 그대로)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Always reply in Korean unless the user explicitly asks you to use another language."

# ✅ 입력 검증/캐스팅용 스펙
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
    """
    data에서 key를 읽어 스펙대로 캐스팅 + 클램핑.
    실패하면 DEFAULT_PARAMS[key]로 복귀.
    """
    default = DEFAULT_PARAMS[key]
    spec = _PARAM_SPECS[key]
    raw = data.get(key, default)

    try:
        if spec["type"] is int:
            # "50.0" 같은 문자열도 들어올 수 있어 안전하게 처리
            val = int(float(raw))
        else:
            val = float(raw)
        val = _clamp(val, spec["min"], spec["max"])
        return val
    except Exception:
        return default


def _extract_request_config(data: dict):
    """
    JSON 요청에서 system_prompt/params를 추출.
    - 세션에 저장하지 않음
    - 누락/오염은 DEFAULT로 보정
    """
    sp = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    if not isinstance(sp, str):
        sp = DEFAULT_SYSTEM_PROMPT
    sp = sp.strip()
    if not sp:
        sp = DEFAULT_SYSTEM_PROMPT

    params = {k: _coerce_param(data, k) for k in DEFAULT_PARAMS.keys()}
    return sp, params


@bp.route("/llama", methods=["GET", "POST"])
def llama_chat():
    # =========================
    # ✅ 세션에는 "대화내역만" 유지
    # =========================
    session.setdefault("chat_history", [])

    # =========================
    # ✅ JSON 요청: 채팅 Send (Apply된 값은 JSON에 실려옴)
    # =========================
    if request.method == "POST" and request.is_json:
        data = request.get_json(silent=True) or {}

        # (예전 UI 호환) JSON으로 reset_chat을 보낸 경우도 처리
        if data.get("action") == "reset_chat":
            session["chat_history"] = []
            session.modified = True
            log.info("[RESET_CHAT:JSON] chat_history cleared")
            return jsonify({"ok": True})

        user_input = (data.get("prompt") or "").strip()
        if not user_input:
            return jsonify({"answer": ""})

        system_prompt, params = _extract_request_config(data)

        # ✅ generate 직전 로그(너무 길면 보기 힘드니 앞부분만)
        sp_preview = (system_prompt[:180] + "…") if len(system_prompt) > 180 else system_prompt
        log.info("[SEND] system_prompt=%r", sp_preview)
        log.info("[SEND] params=%s", params)

        # ✅ 대화 히스토리 누적(세션)
        session["chat_history"].append({"role": "user", "content": user_input})

        assistant_reply = generate_chat(
            session["chat_history"],
            system_prompt=system_prompt,
            **params
        )

        session["chat_history"].append({"role": "assistant", "content": assistant_reply})
        session.modified = True
        return jsonify({"answer": assistant_reply})

    # =========================
    # ✅ FORM 요청: Reset Chat만 지원
    # (Apply는 이제 클라이언트에서만 처리하므로 서버에서 저장/적용 X)
    # =========================
    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "reset_chat":
            session["chat_history"] = []
            session.modified = True
            log.info("[RESET_CHAT:FORM] chat_history cleared")
            return redirect(url_for("llama.llama_chat"))

        # 알 수 없는 action이 와도 화면 복귀
        return redirect(url_for("llama.llama_chat"))

    # =========================
    # ✅ GET: 항상 "디폴트 값"으로 렌더링
    # (쿠키/세션에 저장된 system_prompt/params는 아예 사용하지 않음)
    # =========================
    return render_template(
        "llama/llama.html",
        chat_history=session.get("chat_history", []),
        params=DEFAULT_PARAMS,
        system_prompt=DEFAULT_SYSTEM_PROMPT
    )
