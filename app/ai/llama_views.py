# llama_views.py
import logging
from flask import Blueprint, render_template, request, session, jsonify
from .llama_model import generate_chat

bp = Blueprint("llama", __name__)

log = logging.getLogger("llama")
log.setLevel(logging.INFO)


# (원래 DEFAULT들 그대로...)

@bp.route("/llama", methods=["GET", "POST"])
def llama_chat():
    session.setdefault("chat_history", [])
    session.setdefault("params", DEFAULT_PARAMS.copy())
    session.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)

    # ✅ JSON 요청(채팅 Send)
    if request.method == "POST" and request.is_json:
        data = request.get_json()
        user_input = data.get("prompt", "").strip()
        if not user_input:
            return jsonify({"answer": ""})

        # (system_prompt를 JSON에서 안 받는 구조면 이 부분은 없어도 됨)
        if "system_prompt" in data:
            session["system_prompt"] = data["system_prompt"]

        # ✅ generate 직전에 "이번 요청에 사용된 값" 로그
        log.info("[SEND] system_prompt=%r", session["system_prompt"])
        log.info("[SEND] params=%s", session["params"])

        session["chat_history"].append({"role": "user", "content": user_input})

        assistant_reply = generate_chat(
            session["chat_history"],
            system_prompt=session["system_prompt"],
            **session["params"]
        )

        session["chat_history"].append({"role": "assistant", "content": assistant_reply})
        session.modified = True
        return jsonify({"answer": assistant_reply})

    # ✅ FORM 요청(Apply)
    if request.method == "POST":
        action = request.form.get("action")

        if action == "apply_params":
            for key in session["params"]:
                if key in request.form:
                    v = request.form[key]
                    session["params"][key] = float(v) if "." in v else int(v)

            if "system_prompt" in request.form:
                session["system_prompt"] = request.form["system_prompt"]

            # ✅ Apply 직후 저장된 값 로그
            log.info("[APPLY] system_prompt=%r", session["system_prompt"])
            log.info("[APPLY] params=%s", session["params"])

        session.modified = True

    return render_template(
        "llama/llama.html",
        chat_history=session["chat_history"],
        params=session["params"],
        system_prompt=session["system_prompt"]
    )
