import unittest

from research_app.domain import SourceType
from research_app.rag.chunking import chunk_document, chunk_id_for, split_text
from tests.rag_helpers import doc


class SplitTextTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_text("hello world", 200, 20), ["hello world"])

    def test_empty_text_has_no_chunks(self):
        self.assertEqual(split_text("   ", 200, 20), [])

    def test_chunks_respect_size_and_overlap(self):
        text = " ".join(f"word{i}" for i in range(300))
        pieces = split_text(text, 200, 40)
        self.assertGreater(len(pieces), 3)
        self.assertTrue(all(len(p) <= 200 for p in pieces))
        # consecutive chunks share text (overlap) and together cover everything
        self.assertTrue(any(a[-15:] in b for a, b in zip(pieces, pieces[1:])))
        self.assertIn("word299", pieces[-1])

    def test_prefers_paragraph_boundary(self):
        text = "a" * 120 + "\n\n" + "b" * 120
        self.assertEqual(split_text(text, 200, 0)[0], "a" * 120)

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            split_text("text", 100, 100)


class ChunkDocumentTests(unittest.TestCase):
    def test_metadata_is_carried_and_ids_are_deterministic(self):
        d = doc("https://docs.example.com/a", "x " * 900, source_type=SourceType.OFFICIAL_DOCS,
                technology="Example")
        first = chunk_document(d, 500, 50)
        again = chunk_document(d, 500, 50)
        self.assertGreater(len(first), 1)
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in again])
        self.assertEqual(len({c.chunk_id for c in first}), len(first))
        self.assertEqual(first[0].chunk_id, chunk_id_for(d.source_id or "", 0))
        self.assertEqual(first[0].source_type, SourceType.OFFICIAL_DOCS)
        self.assertEqual(first[0].domain, "docs.example.com")
        self.assertEqual(first[0].technology, "Example")

    def test_falls_back_to_snippet_and_skips_empty(self):
        with_snippet = doc("https://a.example.com/x", "")
        with_snippet.content = None
        with_snippet.snippet = "only a snippet"
        self.assertEqual([c.text for c in chunk_document(with_snippet, 500, 50)], ["only a snippet"])
        empty = doc("https://a.example.com/y", "  ")
        self.assertEqual(chunk_document(empty, 500, 50), [])


if __name__ == "__main__":
    unittest.main()
