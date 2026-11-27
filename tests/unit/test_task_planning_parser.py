from task_planning.planner import parse_response


def _raises(fn, exc_type=ValueError) -> bool:
    try:
        fn()
    except exc_type:
        return True
    return False


def test_parses_thought_and_receptacle_id():
    response = (
        'THOUGHT: The mug is listed inside desk_0, so navigate there.\n'
        'CALL: tool_navigate_to_receptacle("desk_0")'
    )
    thought, receptacle_id = parse_response(response, valid_receptacle_ids={"desk_0", "shelf_1"})
    assert receptacle_id == "desk_0"
    assert "mug" in thought


def test_multiline_thought_is_captured_up_to_call():
    response = (
        "THOUGHT: Two receptacles contain a cup: desk_0 and shelf_1.\n"
        "No colour/location was given, so I default to the desk.\n"
        'CALL: tool_navigate_to_receptacle("desk_0")'
    )
    thought, receptacle_id = parse_response(response, valid_receptacle_ids={"desk_0", "shelf_1"})
    assert receptacle_id == "desk_0"
    assert "Two receptacles" in thought
    assert "default to the desk" in thought


def test_namespaced_call_is_parsed():
    # Gemini namespaces the call when echoing it back in text with automatic
    # function calling enabled, e.g. "default_api.tool_navigate_to_..." --
    # seen in a real run, not hypothetical.
    response = 'CALL: default_api.tool_navigate_to_receptacle("floor_0")'
    thought, receptacle_id = parse_response(response, valid_receptacle_ids={"floor_0"})
    assert receptacle_id == "floor_0"
    assert thought == ""  # no THOUGHT: line in this response, and that's fine


def test_keyword_argument_call_is_parsed():
    # Also seen in a real run: keyword arg instead of positional, single quotes.
    response = "CALL: tool_navigate_to_receptacle(receptacle_id='forklift_0')"
    thought, receptacle_id = parse_response(response, valid_receptacle_ids={"forklift_0"})
    assert receptacle_id == "forklift_0"


def test_missing_call_line_raises():
    assert _raises(lambda: parse_response("THOUGHT: I don't know what to do.", valid_receptacle_ids={"desk_0"}))


def test_hallucinated_receptacle_id_raises():
    response = 'THOUGHT: made this up.\nCALL: tool_navigate_to_receptacle("nonexistent_99")'
    assert _raises(lambda: parse_response(response, valid_receptacle_ids={"desk_0", "shelf_1"}))
