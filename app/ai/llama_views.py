from flask import Blueprint, render_template, request, session
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


@bp.route("/llama", methods=["GET", "POST"])
def llama_chat():
    if "chat_history" not in session:
        session["chat_history"] = []

    if "params" not in session:
        session["params"] = DEFAULT_PARAMS.copy()

    if request.method == "POST":
        action = request.form.get("action")

        # ✅ 하이퍼파라미터 적용
        if action == "apply_params":
            for key in session["params"]:
                if key in request.form:
                    session["params"][key] = float(request.form[key]) \
                        if "." in request.form[key] else int(request.form[key])

        # ✅ 메시지 전송
        elif action == "send_message":
            user_input = request.form["prompt"]

            session["chat_history"].append({
                "role": "user",
                "content": user_input
            })

            assistant_reply = generate_chat(
                session["chat_history"],
                **session["params"]
            )

            session["chat_history"].append({
                "role": "assistant",
                "content": assistant_reply
            })

        session.modified = True

    return render_template(
        "llama/llama.html",
        chat_history=session["chat_history"],
        params=session["params"]
    )
