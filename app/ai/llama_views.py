# llama_views.py
import logging
from flask import Blueprint, render_template, request, session, jsonify
from .llama_model import generate_chat

bp = Blueprint("llama", __name__)

# ✅ 터미널에 안 찍히는 경우 대비: handler를 직접 추가
log = logging.getLogger("llama")
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
log.setLevel(logging.INFO)

# ✅ 기본 하이퍼파라미터 (원래 있던 거)
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
    "repetition_penalty": 1.1,
}

# ✅ 기본 System Prompt (원래 있던 거)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@bp.route("/llama", methods=["GET", "POST"])
def llama_chat():
    # =========================
    # 세션 초기화
    # =========================
    session.setdefault("chat_history", [])
    session.setdefault("params", DEFAULT_PARAMS.copy())
    session.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)

    # =========================
    # ✅ JSON 요청(채팅 Send)
    # =========================
    if request.method == "POST" and request.is_json:
        data = request.get_json(silent=True) or {}

        user_input = (data.get("prompt") or "").strip()
        if not user_input:
            return jsonify({"answer": ""})

        # (네 HTML fetch에서 system_prompt 제거했으면 이 블록은 없어도 됨)
        if "system_prompt" in data and isinstance(data["system_prompt"], str):
            session["system_prompt"] = data["system_prompt"]

        # ✅ generate 직전에 이번 요청에 사용될 값 확인
        log.info("[SEND] system_prompt=%r", session.get("system_prompt"))
        log.info("[SEND] params=%s", session.get("params"))

        session["chat_history"].append({"role": "user", "content": user_input})

        assistant_reply = generate_chat(
            session["chat_history"],
            system_prompt=session["system_prompt"],
            **session["params"]
        )

        session["chat_history"].append({"role": "assistant", "content": assistant_reply})
        session.modified = True
        return jsonify({"answer": assistant_reply})

    # =========================
    # ✅ FORM 요청(Apply)
    # =========================
    if request.method == "POST":
        action = request.form.get("action")

        if action == "apply_params":
            # params 업데이트
            for key in session["params"]:
                if key in request.form:
                    v = request.form[key]
                    session["params"][key] = float(v) if "." in v else int(v)

            # system prompt 업데이트
            if "system_prompt" in request.form:
                session["system_prompt"] = request.form["system_prompt"]

            # ✅ Apply 직후 저장된 값 확인
            log.info("[APPLY] system_prompt=%r", session.get("system_prompt"))
            log.info("[APPLY] params=%s", session.get("params"))

        session.modified = True

    # =========================
    # GET / 화면 렌더
    # =========================
    return render_template(
        "llama/llama.html",
        chat_history=session["chat_history"],
        params=session["params"],
        system_prompt=session["system_prompt"]
    )
