from multimodal_retrieval_agent.intent.rule_based import classify_intent


def test_text_to_image_topk():
    result = classify_intent("Please show me three chair photos")
    assert result.needs_retrieval == 1
    assert result.query_type == "text_to_image"
    assert result.category == "chair"
    assert result.search_mode == "top_k"
    assert result.top_k == 3


def test_no_retrieval():
    result = classify_intent("What is the weather in London?")
    assert result.needs_retrieval == 0
