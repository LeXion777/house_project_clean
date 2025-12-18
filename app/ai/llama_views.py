from flask import Blueprint, render_template, request, session
from .llama_model import generate_chat

bp = Blueprint("llama", __name__)

# 기본 하이퍼파라미터
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
    "repetition_penalty": 1.1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
}

# 기본 System Prompt
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@bp.route("/llama", methods=["GET", "POST"])
def llama_chat():
    # =========================
    # 세션 초기화
    # =========================
    if "chat_history" not in session:
        session["chat_history"] = []

    if "params" not in session:
        session["params"] = DEFAULT_PARAMS.copy()

    if "system_prompt" not in session:
        session["system_prompt"] = DEFAULT_SYSTEM_PROMPT

    # =========================
    # POST 처리
    # =========================
    if request.method == "POST":
        action = request.form.get("action")

        # -------------------------
        # 1️⃣ 하이퍼파라미터 / 시스템 프롬프트 적용
        # -------------------------
        if action == "apply_params":
            # 하이퍼파라미터 저장
            for key in session["params"]:
                if key in request.form:
                    value = request.form[key]
                    session["params"][key] = (
                        float(value) if "." in value else int(value)
                    )

            # System Prompt 저장
            if "system_prompt" in request.form:
                session["system_prompt"] = request.form["system_prompt"]

        # -------------------------
        # 2️⃣ 메시지 전송
        # -------------------------
        elif action == "send_message":
            user_input = request.form.get("prompt", "").strip()

            if user_input:
                session["chat_history"].append({
                    "role": "user",
                    "content": user_input
                })

                assistant_reply = generate_chat(
                    session["chat_history"],
                    system_prompt=session["system_prompt"],  # 🔥 핵심
                    **session["params"]
                )

                session["chat_history"].append({
                    "role": "assistant",
                    "content": assistant_reply
                })

        session.modified = True

    # =========================
    # 렌더링
    # =========================
    return render_template(
        "llama/llama.html",
        chat_history=session["chat_history"],
        params=session["params"],
        system_prompt=session["system_prompt"]
    )
