import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from paper_radar.models import Paper
from paper_radar.recommender import AIRecommender, _extract_json


class FakeStream:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self.events = events

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.events)


class FakeResponses:
    def __init__(self) -> None:
        self.request = None

    def create(self, **request: object) -> FakeStream:
        self.request = request
        response = SimpleNamespace(status="completed", output_text='{"evaluations": []}')
        return FakeStream(
            [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta='{"evaluations":',
                ),
                SimpleNamespace(type="response.output_text.delta", delta=" []}"),
                SimpleNamespace(type="response.completed", response=response),
            ]
        )


def make_recommender() -> AIRecommender:
    recommender = object.__new__(AIRecommender)
    recommender.model = "gpt-5.6-sol"
    recommender.reasoning_effort = "high"
    recommender.prompt_template = "Evaluate these papers:\n\n{PAPER_INFO}"
    return recommender


def make_paper() -> Paper:
    now = datetime.now(UTC)
    return Paper(
        paper_id="2508.00001",
        title="Excitons in a two-dimensional semiconductor",
        authors=("Test Author",),
        abstract="We study excitons in a monolayer semiconductor.",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2508.00001",
        pdf_url="https://arxiv.org/pdf/2508.00001",
    )


def valid_evaluation() -> dict[str, object]:
    return {
        "paper_id": "2508.00001",
        "score": 3,
        "reason": "Directly relevant to two-dimensional exciton research.",
        "key_relevance": ["two-dimensional semiconductor", "exciton"],
        "title_zh": "二维半导体中的激子",
        "summary_zh": "本文研究单层半导体中的激子性质。该工作与二维材料光学直接相关。",
    }


def test_request_uses_streaming_message_input() -> None:
    responses = FakeResponses()
    recommender = make_recommender()
    recommender.client = SimpleNamespace(responses=responses)

    result = recommender._request("Evaluate this paper", structured=True)

    assert result == '{"evaluations": []}'
    assert responses.request is not None
    assert responses.request["stream"] is True
    assert responses.request["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Evaluate this paper"}],
        }
    ]
    assert responses.request["reasoning"] == {"effort": "high"}
    assert responses.request["text"]["format"]["type"] == "json_schema"


def test_request_uses_json_mode_for_retry() -> None:
    responses = FakeResponses()
    recommender = make_recommender()
    recommender.client = SimpleNamespace(responses=responses)

    recommender._request("Return JSON", structured=False)

    assert responses.request is not None
    assert responses.request["text"] == {"format": {"type": "json_object"}}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"evaluations": []}', {"evaluations": []}),
        ("```json\n{\"evaluations\": []}\n```", {"evaluations": []}),
        ("Result: {\"evaluations\": []}", {"evaluations": []}),
        ("[]", {"evaluations": []}),
        ('"{\\"evaluations\\": []}"', {"evaluations": []}),
    ],
)
def test_extract_json_accepts_relay_variants(
    response: str, expected: dict[str, object]
) -> None:
    assert _extract_json(response) == expected


def test_evaluate_retries_invalid_structured_output_with_json_mode() -> None:
    recommender = make_recommender()
    outputs = iter(["", json.dumps({"evaluations": [valid_evaluation()]})])
    request_modes: list[bool] = []

    def fake_request(prompt: str, *, structured: bool) -> str:
        request_modes.append(structured)
        return next(outputs)

    recommender._request = fake_request
    result = recommender.evaluate([make_paper()], {})[0]

    assert request_modes == [True, False]
    assert result.used_ai is True
    assert result.title_zh == "二维半导体中的激子"
    assert result.summary_zh


def test_evaluate_retries_incomplete_evaluation() -> None:
    recommender = make_recommender()
    incomplete = valid_evaluation()
    incomplete["summary_zh"] = ""
    outputs = iter(
        [
            json.dumps({"evaluations": [incomplete]}),
            json.dumps([valid_evaluation()]),
        ]
    )
    request_modes: list[bool] = []

    def fake_request(prompt: str, *, structured: bool) -> str:
        request_modes.append(structured)
        return next(outputs)

    recommender._request = fake_request
    result = recommender.evaluate([make_paper()], {})[0]

    assert request_modes == [True, False]
    assert result.used_ai is True
    assert result.key_relevance == ("two-dimensional semiconductor", "exciton")


def test_evaluate_accepts_relay_field_aliases() -> None:
    recommender = make_recommender()
    relay_evaluation = valid_evaluation()
    relay_evaluation["key_relevance_items"] = relay_evaluation.pop("key_relevance")
    relay_evaluation["chinese_title"] = relay_evaluation.pop("title_zh")
    relay_evaluation["chinese_summary"] = relay_evaluation.pop("summary_zh")
    recommender._request = lambda prompt, *, structured: json.dumps(
        {"evaluations": [relay_evaluation]}
    )

    result = recommender.evaluate([make_paper()], {})[0]

    assert result.used_ai is True
    assert result.title_zh == "二维半导体中的激子"
    assert result.summary_zh
    assert result.key_relevance == ("two-dimensional semiconductor", "exciton")


def test_evaluate_rejects_incomplete_json_mode_result() -> None:
    recommender = make_recommender()
    recommender._request = lambda prompt, *, structured: '{"evaluations": []}'

    with pytest.raises(ValueError, match="1 paper evaluation"):
        recommender.evaluate([make_paper()], {})
