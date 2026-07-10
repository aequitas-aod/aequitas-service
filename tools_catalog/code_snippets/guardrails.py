import json
from langchain_classic.evaluation import JsonValidityEvaluator
from langchain_classic.evaluation import JsonEqualityEvaluator
from langchain_classic.evaluation import RegexMatchStringEvaluator
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately


def json_format_validator(json_string):
    """
    JSON Format Validator
    Checks if the generated response is valid JSON
    """
    evaluator = JsonValidityEvaluator()
    # '{"x": 1}'# correct '{x: 1}'  incorrect
    eval_result = evaluator.evaluate_strings(prediction=json_string)
    # {'score': 0, 'reasoning': 'Expecting property name enclosed in double quotes: line 1 column 2 (char 1)'}
    return eval_result


def json_equality_evaluator(prediction_json):
    """
    JsonEqualityEvaluator
    Checks the equality of JSONs after parsing (the order of keys in JSON does not matter)
    example: prediction_json = {"a":1,"b":[2,3]}
    """
    evaluator = JsonEqualityEvaluator()
    eval_result = evaluator.evaluate_strings(
        prediction=prediction_json,
        reference='{"b":[2,3],"a":2}',
    )
    return eval_result


def regex_match_evaluator(prediction):
    """
    RegexMatchEvaluator
    Checks for a match against a regular expression
    example: prediction = "Order ID: ABC-1234"
    """
    evaluator = RegexMatchStringEvaluator()
    result = evaluator.evaluate_strings(
        prediction=prediction,
        reference=r"^Order ID: [A-Z]{3}-\d{4}$",
    )
    return result["score"]


def token_limit(conversation_history):
    """
    Token Limit
    Tracking and pruning history to a token limit to avoid exceeding model context.
    """
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=conversation_history),
    ]
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=256,
        start_on="human",
        include_system=True,
    )
    return trimmed
