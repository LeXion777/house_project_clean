from flask import Blueprint, request, jsonify
from .models import (
    qa_pipeline,
    textgen_pipeline,
    translate_pipeline,
    sentiment_pipeline,
    ner_pipeline
)

bp = Blueprint("ai", __name__)


@bp.route("/ai-chat", methods=["POST"])
def ai_chat():
    data = request.get_json()
    question = data.get("question", "")
    model_type = data.get("model", "qa")

    base_context = """
    서울은 대한민국의 수도이며 정치, 경제, 문화 중심지이다.
    대표적인 관광지는 광화문, 경복궁, 남산타워 등이 있다.
    """

    conversation_context = data.get("conversation", "") or ""
    merged_context = base_context.strip() + "\n" + conversation_context.strip()

    # ---------------- QA ----------------
    if model_type == "qa":
        result = qa_pipeline(
            question=question,
            context=merged_context
        )
        return jsonify({"answer": result["answer"]})

    # ---------------- TextGen ----------------
    elif model_type == "textgen":
        output = textgen_pipeline(
            question,
            max_length=80,
            do_sample=True,
            top_p=0.92
        )
        return jsonify({"answer": output[0]["generated_text"]})

    # ---------------- Translation ----------------
    elif model_type == "translate":
        result = translate_pipeline(question)
        return jsonify({"answer": result[0]["translation_text"]})

    # ---------------- Sentiment ----------------
    elif model_type == "sentiment":
        result = sentiment_pipeline(question)[0]
        label = result["label"]
        score = round(float(result["score"]) * 100, 1)

        sentiment = "긍정 😊" if label in ["POSITIVE", "LABEL_1"] else "부정 😞"
        return jsonify({
            "answer": f"감정 분석: {sentiment}\n신뢰도: {score}%"
        })

    # ---------------- NER ----------------
    elif model_type == "ner":
        entities = ner_pipeline(question)

        if not entities:
            return jsonify({"answer": "인식된 개체명이 없습니다."})

        lines = ["🔎 인식된 개체명:"]
        for e in entities:
            lines.append(
                f"- {e['word']} ({e['entity_group']}, {round(e['score']*100,1)}%)"
            )

        return jsonify({"answer": "<br>".join(lines)})

    return jsonify({"answer": "지원하지 않는 모델입니다."})
