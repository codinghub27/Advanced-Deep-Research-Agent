import unittest

from research_app.domain import Complexity, ResearchTask, RunStatus, SourceType
from research_app.domain.legacy import (
    CACHE_HIT_PLACEHOLDER,
    query_from_state,
    run_from_state,
    source_from_legacy,
    source_to_legacy,
    sources_from_legacy,
    tasks_from_sub_questions,
    tasks_to_sub_questions,
)

# Shapes copied from what simple_search_node / search_node / main.py produce.
LEGACY_SOURCES = [
    {"url": "https://example.com/a", "title": "Alpha", "snippet": "first snippet, cut at 150 chars "},
    {"url": "https://example.org/b", "title": "", "snippet": ""},
]


class SourceRoundTripTests(unittest.TestCase):
    def test_legacy_source_round_trips_exactly(self):
        for item in LEGACY_SOURCES:
            with self.subTest(url=item["url"]):
                self.assertEqual(source_to_legacy(source_from_legacy(item)), item)

    def test_sse_shape_has_exactly_three_keys(self):
        doc = source_from_legacy(LEGACY_SOURCES[0], source_type=SourceType.OFFICIAL_DOCS, task_id="t", query="q")
        self.assertEqual(set(source_to_legacy(doc)), {"url", "title", "snippet"})

    def test_legacy_source_carries_no_content(self):
        self.assertIsNone(source_from_legacy(LEGACY_SOURCES[0]).content)

    def test_bulk_conversion_skips_invalid_and_dedupes(self):
        items = [
            LEGACY_SOURCES[0],
            {"url": "https://EXAMPLE.com/a/", "title": "dup", "snippet": ""},
            {"url": "", "title": "no url", "snippet": ""},
            {"url": "javascript:alert(1)", "title": "bad", "snippet": ""},
            None,
            LEGACY_SOURCES[1],
        ]
        docs = sources_from_legacy(items)
        self.assertEqual([d.url for d in docs], ["https://example.com/a", "https://example.org/b"])


class QueryAndTaskTests(unittest.TestCase):
    STATE = {
        "question": "Compare laptops",
        "is_simple": False,
        "history": [{"question": "q1", "answer": "a1"}, {"question": "broken"}, "junk"],
    }

    def test_complexity_only_read_when_classified(self):
        self.assertIsNone(query_from_state(self.STATE).complexity)
        self.assertEqual(query_from_state(self.STATE, classified=True).complexity, Complexity.COMPLEX)
        simple = query_from_state({"question": "x", "is_simple": True}, classified=True)
        self.assertEqual(simple.complexity, Complexity.SIMPLE)

    def test_history_keeps_valid_turns_only(self):
        q = query_from_state(self.STATE, user_id=7, session_id=3)
        self.assertEqual([(t.question, t.answer) for t in q.history], [("q1", "a1")])
        self.assertEqual((q.text, q.user_id, q.session_id), ("Compare laptops", 7, 3))

    def test_sub_questions_round_trip_in_order(self):
        subs = ["q one", "q two", "q three"]
        tasks = tasks_from_sub_questions(subs)
        self.assertTrue(all(isinstance(t, ResearchTask) for t in tasks))
        self.assertEqual(tasks_to_sub_questions(tasks), subs)

    def test_cache_placeholder_and_blank_entries_are_not_tasks(self):
        self.assertEqual(tasks_from_sub_questions([CACHE_HIT_PLACEHOLDER, "", "  ", "real"]).__len__(), 1)
        self.assertEqual(tasks_from_sub_questions([CACHE_HIT_PLACEHOLDER]), [])

    def test_placeholder_matches_what_the_graph_emits(self):
        # Guards against the sentinel drifting in state.py (read as text; no import).
        with open("research_app/agent/state.py", encoding="utf-8") as fh:
            self.assertIn(f'"{CACHE_HIT_PLACEHOLDER}"', fh.read())


class RunFromStateTests(unittest.TestCase):
    def test_finished_research_state(self):
        state = {
            "question": "Compare laptops",
            "final_answer": "A long report",
            "sub_questions": ["q1", "q2"],
            "sources": LEGACY_SOURCES + [LEGACY_SOURCES[0]],
            "cache_hit": False,
            "is_simple": False,
            "history": [],
        }
        run = run_from_state(state, user_id=1, session_id=2, run_id="abc")
        self.assertEqual(run.run_id, "abc")
        self.assertEqual(run.status, RunStatus.COMPLETED)
        self.assertEqual(run.final_answer, "A long report")
        self.assertEqual(len(run.tasks), 2)
        self.assertEqual(len(run.sources), 2)  # duplicate URL collapsed
        self.assertEqual(run.query.complexity, Complexity.COMPLEX)
        self.assertFalse(run.cache_hit)

    def test_cache_hit_state(self):
        state = {
            "question": "Compare laptops",
            "final_answer": "cached",
            "sub_questions": [CACHE_HIT_PLACEHOLDER],
            "sources": [],
            "cache_hit": True,
            "is_simple": False,
        }
        run = run_from_state(state)
        self.assertTrue(run.cache_hit)
        self.assertEqual((run.tasks, run.sources), ([], []))
        self.assertIsNone(run.query.complexity)  # never classified on a cache hit

    def test_run_id_is_generated_when_not_given(self):
        self.assertTrue(run_from_state({"question": "x"}).run_id)


if __name__ == "__main__":
    unittest.main()
