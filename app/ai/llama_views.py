from flask import Blueprint, render_template, request, session
from .llama_model import generate_chat

bp = Blueprint("llama", __name__)


@bp.route("/llama", methods=["GET", "POST"])
def llama_chat():
    if "chat_history" not in session:
        session["chat_history"] = []

    params = {
        "temperature": session.get("temperature", 0.7),
        "top_p": session.get("top_p", 0.9),
        "max_tokens": session.get("max_tokens", 512),
        "repetition_penalty": session.get("repetition_penalty", 1.1)
    }

    if request.method == "POST":
        user_input = request.form["prompt"]

        params["temperature"] = float(request.form["temperature"])
        params["top_p"] = float(request.form["top_p"])
        params["max_tokens"] = int(request.form["max_tokens"])
        params["repetition_penalty"] = float(request.form["repetition_penalty"])

        session.update(params)

        session["chat_history"].append({
            "role": "user",
            "content": user_input
        })

        assistant_reply = generate_chat(
            session["chat_history"],
            **params
        )

        session["chat_history"].append({
            "role": "assistant",
            "content": assistant_reply
        })

        session.modified = True
    return render_template("llama/llama.html", chat_history=session["chat_history"], **params)

