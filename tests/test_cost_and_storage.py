"""Cost accounting and the conversation storage implementations."""

from __future__ import annotations

import time

import pytest

from services import cost_service
from services.cost_service import Usage, compute_cost, pricing_for, usage_from_response
from services.storage.base import Message, UsageRecord
from services.storage.memory_store import InMemoryConversationStore
from services.storage.sql_store import SqlConversationStore


class FakeResponse:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


class TestPricing:
    def test_known_model(self):
        assert pricing_for("gpt-5.4-mini")["input"] == 0.25

    def test_prefix_match_picks_longest(self):
        assert pricing_for("gpt-5.4-mini-2026-01-01") == pricing_for("gpt-5.4-mini")

    def test_unknown_model_falls_back(self):
        rates = pricing_for("some-unreleased-model")
        assert rates["input"] > 0 and rates["output"] > 0

    def test_cached_tokens_are_cheaper(self):
        full = compute_cost("gpt-5.4-mini", input_tokens=1_000_000, output_tokens=0)
        cached = compute_cost(
            "gpt-5.4-mini", input_tokens=1_000_000, cached_input_tokens=1_000_000
        )
        assert cached < full
        assert cached == pytest.approx(0.025)

    def test_cost_is_zero_for_no_usage(self):
        assert compute_cost("gpt-5.4-mini", input_tokens=0) == 0.0


class TestUsageExtraction:
    def test_reads_langchain_usage_metadata(self):
        response = FakeResponse(
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "input_token_details": {"cache_read": 800},
                "output_token_details": {"reasoning": 50},
            }
        )
        usage = usage_from_response(response, "gpt-5.4-mini")
        assert usage.input_tokens == 1000
        assert usage.cached_input_tokens == 800
        assert usage.reasoning_tokens == 50
        assert usage.cost_usd > 0

    def test_reads_openai_style_metadata(self):
        response = FakeResponse(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 400},
                }
            }
        )
        usage = usage_from_response(response, "gpt-5.4-mini")
        assert usage.input_tokens == 500
        assert usage.cached_input_tokens == 400

    def test_missing_usage_is_not_fatal(self):
        usage = usage_from_response(FakeResponse(), "gpt-5.4-mini")
        assert usage.input_tokens == 0 and usage.cost_usd == 0.0

    def test_cached_never_exceeds_input(self):
        response = FakeResponse(
            usage_metadata={"input_tokens": 100, "output_tokens": 0,
                            "input_token_details": {"cache_read": 999}}
        )
        assert usage_from_response(response, "gpt-5.4-mini").cached_input_tokens == 100

    def test_usage_adds(self):
        total = Usage(model="m", input_tokens=100, output_tokens=10, cost_usd=0.001, requests=1)
        total.add(Usage(model="m", input_tokens=50, output_tokens=5, cost_usd=0.002, requests=1))
        assert total.input_tokens == 150 and total.requests == 2
        assert total.cost_usd == pytest.approx(0.003)

    def test_cache_hit_rate(self):
        usage = Usage(input_tokens=1000, cached_input_tokens=750)
        assert usage.cache_hit_rate == 0.75
        assert usage.uncached_input_tokens == 250


class TestTokenEstimation:
    def test_estimates_text(self):
        assert cost_service.estimate_tokens("hello world") > 0

    def test_empty_is_zero(self):
        assert cost_service.estimate_tokens("") == 0

    def test_longer_text_costs_more(self):
        short = cost_service.estimate_tokens("a b c")
        long = cost_service.estimate_tokens("a b c " * 200)
        assert long > short

    def test_estimates_messages(self):
        from langchain_core.messages import HumanMessage, SystemMessage

        total = cost_service.estimate_messages_tokens(
            [SystemMessage(content="You are a bot."), HumanMessage(content="Hello there")]
        )
        assert total > 0


def _store_contract(store):
    """Behaviour every ConversationStore implementation must satisfy."""
    store.append(Message(thread_id="t1", role="user", content="first question"))
    store.append(Message(thread_id="t1", role="assistant", content="an answer"))

    history = store.history("t1")
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[0].content == "first question"

    assert store.history("unknown-thread") == []

    threads = store.threads()
    assert threads[0]["thread_id"] == "t1"
    assert threads[0]["message_count"] == 2

    store.record_usage(UsageRecord(
        thread_id="t1", user_id="u1", model="m",
        input_tokens=100, cached_input_tokens=50, output_tokens=20,
        reasoning_tokens=0, cost_usd=0.5,
    ))
    assert store.cost_since(user_id="u1", since=time.time() - 60) == pytest.approx(0.5)
    assert store.cost_since(user_id="u2", since=time.time() - 60) == 0.0

    assert store.delete_thread("t1") is True
    assert store.history("t1") == []
    assert store.delete_thread("t1") is False


class TestInMemoryStore:
    def test_satisfies_the_contract(self):
        _store_contract(InMemoryConversationStore())

    def test_bounds_message_count(self):
        store = InMemoryConversationStore(max_messages=4)
        for index in range(10):
            store.append(Message(thread_id="t", role="user", content=f"message {index}"))
        history = store.history("t")
        assert len(history) == 4
        assert history[-1].content == "message 9"   # newest kept

    def test_bounds_total_characters(self):
        store = InMemoryConversationStore(max_messages=100, max_chars=500)
        for index in range(20):
            store.append(Message(thread_id="t", role="user", content="x" * 100))
        assert sum(len(m.content) for m in store.history("t")) <= 500

    def test_evicts_oldest_threads(self):
        store = InMemoryConversationStore(max_threads=3)
        for index in range(5):
            store.append(Message(thread_id=f"t{index}", role="user", content="hi"))
        assert store.history("t0") == []
        assert store.history("t4") != []


class TestSqlStore:
    def test_satisfies_the_contract(self, tmp_path):
        _store_contract(SqlConversationStore(f"sqlite:///{tmp_path / 'test.db'}"))

    def test_persists_across_instances(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'persist.db'}"
        first = SqlConversationStore(url)
        first.append(Message(thread_id="t1", role="user", content="remembered"))
        first.close()

        assert SqlConversationStore(url).history("t1")[0].content == "remembered"

    def test_bounds_message_count(self, tmp_path):
        store = SqlConversationStore(f"sqlite:///{tmp_path / 'bounded.db'}", max_messages=3)
        for index in range(8):
            store.append(Message(thread_id="t", role="user", content=f"m{index}"))
        assert len(store.history("t")) == 3

    def test_preserves_metadata(self, tmp_path):
        store = SqlConversationStore(f"sqlite:///{tmp_path / 'meta.db'}")
        store.append(Message(
            thread_id="t", role="assistant", content="answer",
            metadata={"citations": [{"path": "app.py", "start_line": 1}], "user_id": "u1"},
        ))
        assert store.history("t")[0].metadata["citations"][0]["path"] == "app.py"

    def test_isolates_usage_by_user(self, tmp_path):
        store = SqlConversationStore(f"sqlite:///{tmp_path / 'usage.db'}")
        for user, cost in (("u1", 1.0), ("u2", 2.0)):
            store.record_usage(UsageRecord(
                thread_id="t", user_id=user, model="m", input_tokens=1,
                cached_input_tokens=0, output_tokens=1, reasoning_tokens=0, cost_usd=cost,
            ))
        assert store.cost_since(user_id="u1", since=0) == pytest.approx(1.0)
