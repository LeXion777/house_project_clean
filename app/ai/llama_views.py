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
    session.setdefault("chat_history", [])
    session.setdefault("params", DEFAULT_PARAMS.copy())
    session.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)

    # =========================
    # JSON 요청 (fetch)
    # =========================
    if request.method == "POST" and request.is_json:
        data = request.get_json() or {}

        # ✅ 0) 대화 초기화
        if data.get("action") == "reset_chat":
            session["chat_history"] = []
            session.modified = True
            return jsonify({"ok": True})

        # ✅ 1) 메시지 전송
        user_input = (data.get("prompt") or "").strip()
        if not user_input:
            return jsonify({"answer": ""})

        # system prompt 갱신(옵션)
        if "system_prompt" in data:
            session["system_prompt"] = data["system_prompt"]

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
    # FORM 요청 (Apply 등)
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

        # (선택) 폼 방식으로도 초기화하고 싶으면 유지
        elif action == "reset_chat":
            session["chat_history"] = []

        session.modified = True

    return render_template(
        "llama/llama.html",
        chat_history=session["chat_history"],
        params=session["params"],
        system_prompt=session["system_prompt"]
    )
