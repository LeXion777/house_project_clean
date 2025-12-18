from flask import Blueprint, render_template, request, session, jsonify
from .llama_model import generate_chat

bp = Blueprint("llama", __name__)

DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
    "repetition_penalty": 1.1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
}

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
    # 🔥 JSON 요청 (fetch / typing UI)
    # =========================
    if request.method == "POST" and request.is_json:
        data = request.get_json()

        user_input = data.get("prompt", "").strip()
        if not user_input:
            return jsonify({"answer": ""})

        # system prompt 갱신 (옵션)
        if "system_prompt" in data:
            session["system_prompt"] = data["system_prompt"]

        session["chat_history"].append({
            "role": "user",
            "content": user_input
        })

        assistant_reply = generate_chat(
            session["chat_history"],
            system_prompt=session["system_prompt"],
            **session["params"]
        )

        session["chat_history"].append({
            "role": "assistant",
            "content": assistant_reply
        })

        session.modified = True
        return jsonify({"answer": assistant_reply})

    # =========================
    # 기존 FORM 요청 처리
    # =========================
    if request.method == "POST":
        action = request.form.get("action")

        if action == "apply_params":
            for key in session["params"]:
                if key in request.form:
                    v = request.form[key]
                    session["params"][key] = float(v) if "." in v else int(v)

            if "system_prompt" in request.form:
                session["system_prompt"] = request.form["system_prompt"]

        elif action == "send_message":
            user_input = request.form.get("prompt", "").strip()
            if user_input:
                session["chat_history"].append({
                    "role": "user",
                    "content": user_input
                })

                assistant_reply = generate_chat(
                    session["chat_history"],
                    system_prompt=session["system_prompt"],
                    **session["params"]
                )

                session["chat_history"].append({
                    "role": "assistant",
                    "content": assistant_reply
                })

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
