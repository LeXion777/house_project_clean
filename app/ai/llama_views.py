# llama_views.py
import logging
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


@bp.route("/llama", methods=["GET", "POST"])
def llama_chat():
    session.setdefault("chat_history", [])

    if request.method == "POST" and request.is_json:
        data = request.get_json(silent=True) or {}

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
        log.info("[SEND] finetune_id=%s", finetune_id)

        sp_preview = (system_prompt[:180] + "…") if len(system_prompt) > 180 else system_prompt
        log.info("[SEND] system_prompt=%r", sp_preview)
        log.info("[SEND] params=%s", params)

        session["chat_history"].append({"role": "user", "content": user_input})

        # 여기서 finetune_id 전달 → llama_model.py가 필요 시 스위칭 로드
        assistant_reply = generate_chat(
            session["chat_history"],
            system_prompt=system_prompt,
            finetune_id=finetune_id,
            **params,
        )

        session["chat_history"].append({"role": "assistant", "content": assistant_reply})
        session.modified = True
        return jsonify({"answer": assistant_reply})

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "reset_chat":
            session["chat_history"] = []
            session.modified = True
            log.info("[RESET_CHAT:FORM] chat_history cleared")
            return redirect(url_for("llama.llama_chat"))

        return redirect(url_for("llama.llama_chat"))

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
