from types import SimpleNamespace

from paper_radar.recommender import AIRecommender


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


def test_request_uses_streaming_message_input() -> None:
    responses = FakeResponses()
    recommender = object.__new__(AIRecommender)
    recommender.client = SimpleNamespace(responses=responses)
    recommender.model = "gpt-5.6-sol"
    recommender.reasoning_effort = "high"

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
