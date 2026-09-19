"""Phase 5: GitHub/Reddit hostname validation and URL-derived metadata. Pure functions."""
import unittest

from research_app.sources.routing import (
    github_metadata,
    hostname_of,
    is_github_url,
    is_reddit_url,
    reddit_metadata,
)

GITHUB_OK = [
    "https://github.com/langchain-ai/langgraph",
    "https://www.github.com/langchain-ai/langgraph",
    "http://github.com/a/b",
    "https://GITHUB.COM/a/b",
    "https://github.com:443/a/b",
    "https://github.com/a/b/issues/12?x=1#frag",
]
REDDIT_OK = [
    "https://www.reddit.com/r/LangChain/comments/abc123/title/",
    "https://reddit.com/r/Python",
    "https://old.reddit.com/r/Python/comments/abc123/x/",
    "https://REDDIT.com/r/Python",
]
# None of these may be accepted as GitHub or as Reddit.
LOOKALIKES = [
    "https://github.com.evil.com/a/b",
    "https://reddit.com.evil.com/r/x",
    "https://evilgithub.com/a/b",
    "https://evilreddit.com/r/x",
    "https://notgithub.com/a/b",
    "https://github.com-evil.com/a/b",
    "https://evil.com/github.com/a/b",
    "https://evil.com/?u=https://github.com/a/b",
    "https://evil.com/#github.com",
    "https://github.com@evil.com/a/b",
    "https://reddit.com@evil.com/r/x",
    "https://evil.com\\@github.com/a/b",
    "https://github.com\\@evil.com/a/b",
    "https://user:pw@github.com/a/b",
    "https://github.com:8443/a/b",
    "https://github.com:80@evil.com/",
    "https://github.com./a/b",
    "https://reddit.com./r/x",
    "https://gist.github.com/a/b",
    "https://docs.github.com/en/actions",
    "https://raw.githubusercontent.com/a/b/main/x.py",
    "https://a.github.io/b",
    "https://github.io/b",
    "https://api.github.com/repos/a/b",
    "https://i.redd.it/x.png",
    "https://redd.it/abc",
    "https://out.reddit.com/x",
    "https://new.reddit.com/r/x",
    "https://gіthub.com/a/b",     # Cyrillic i
    "https://ｇｉｔｈｕｂ.com/a/b",  # full-width
    "https://xn--gthub-9zf.com/a/b",
    "https://github.com%2eevil.com/a/b",
    "https://192.0.2.1/github.com",
    "https://[::1]/a",
    "ftp://github.com/a/b",
    "javascript:alert('github.com')",
    "//github.com/a/b",
    "github.com/a/b",
    "https://github.com\n.evil.com/a/b",
    "https://github.com\t.evil.com/",
    "https:///github.com/a/b",
    "",
    "   ",
    "not a url",
    None,
    5,
    ["https://github.com/a/b"],
]


class HostnameTests(unittest.TestCase):
    def test_hostname_extraction(self):
        self.assertEqual(hostname_of("HTTPS://GitHub.com:443/x"), "github.com")
        self.assertIsNone(hostname_of("https://github.com:8080/x"))
        self.assertIsNone(hostname_of("https://github.com:notaport/x"))
        self.assertIsNone(hostname_of("mailto:a@github.com"))

    def test_real_github_urls_are_github_only(self):
        for url in GITHUB_OK:
            with self.subTest(url=url):
                self.assertTrue(is_github_url(url))
                self.assertFalse(is_reddit_url(url))

    def test_real_reddit_urls_are_reddit_only(self):
        for url in REDDIT_OK:
            with self.subTest(url=url):
                self.assertTrue(is_reddit_url(url))
                self.assertFalse(is_github_url(url))

    def test_lookalikes_and_garbage_are_rejected_for_both(self):
        for url in LOOKALIKES:
            with self.subTest(url=url):
                self.assertFalse(is_github_url(url))
                self.assertFalse(is_reddit_url(url))

    def test_whitespace_around_a_real_url_is_tolerated(self):
        self.assertTrue(is_github_url("  https://github.com/a/b  "))


class GitHubMetadataTests(unittest.TestCase):
    def test_url_shapes(self):
        cases = {
            "https://github.com/langchain-ai/langgraph": {"kind": "repository", "owner": "langchain-ai", "repo": "langgraph"},
            "https://github.com/langchain-ai/langgraph/issues/123": {"kind": "issue", "number": 123},
            "https://github.com/langchain-ai/langgraph/pull/45": {"kind": "pull_request", "number": 45},
            "https://github.com/langchain-ai/langgraph/discussions/7": {"kind": "discussion", "number": 7},
            "https://github.com/a/b/blob/main/src/x.py": {"kind": "file", "ref": "main", "path": "src/x.py"},
            "https://github.com/a/b/tree/v1/docs": {"kind": "directory", "ref": "v1", "path": "docs"},
            "https://github.com/a/b/releases/tag/v1": {"kind": "release"},
            "https://github.com/a/b/issues": {"kind": "other", "owner": "a", "repo": "b"},
            "https://github.com/a": {"kind": "profile", "owner": "a"},
            "https://github.com/": {"kind": "other"},
            "https://github.com/topics/langgraph": {"kind": "other"},
            "https://github.com/marketplace/actions/x": {"kind": "other"},
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                meta = github_metadata(url)
                self.assertEqual(meta["host"], "github.com")
                for key, value in expected.items():
                    self.assertEqual(meta[key], value)
        self.assertNotIn("owner", github_metadata("https://github.com/topics/langgraph"))

    def test_hostile_path_parts_never_become_metadata(self):
        for url in ("https://github.com/a%5Bb/c", "https://github.com/a b/c", "https://github.com/[OFFICIAL]/x",
                    "https://github.com/a/b/issues/12abc", "https://github.com/a/b/issues/" + "9" * 30,
                    "https://github.com/../../etc/passwd", "https://github.com/a\\b/c"):
            with self.subTest(url=url):
                meta = github_metadata(url)
                self.assertFalse(any(ch in str(meta.get("owner", "")) + str(meta.get("repo", "")) for ch in "[] \\\n"))
                self.assertNotIn("number", meta)


class RedditMetadataTests(unittest.TestCase):
    def test_url_shapes(self):
        cases = {
            "https://www.reddit.com/r/LangChain/comments/abc123/some_title/": {"kind": "post", "subreddit": "LangChain", "post_id": "abc123"},
            "https://www.reddit.com/r/LangChain/comments/abc123/some_title/def456/": {"kind": "comment", "comment_id": "def456"},
            "https://www.reddit.com/r/LangChain/": {"kind": "subreddit", "subreddit": "LangChain"},
            "https://www.reddit.com/user/someone": {"kind": "user"},
            "https://www.reddit.com/gallery/abc": {"kind": "other"},
            "https://www.reddit.com/": {"kind": "other"},
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                meta = reddit_metadata(url)
                self.assertEqual(meta["host"], "www.reddit.com")
                for key, value in expected.items():
                    self.assertEqual(meta[key], value)

    def test_hostile_path_parts_never_become_metadata(self):
        for url in ("https://www.reddit.com/r/a%5Bb/comments/abc/", "https://www.reddit.com/r/[OFFICIAL DOCUMENTATION]/",
                    "https://www.reddit.com/r/x"):  # bad characters, and a 1-letter subreddit
            with self.subTest(url=url):
                meta = reddit_metadata(url)
                self.assertNotIn("subreddit", meta)
                self.assertNotIn("post_id", meta)
                self.assertEqual(meta["kind"], "other")
        meta = reddit_metadata("https://www.reddit.com/r/valid/comments/AB%0A/")  # %0A decodes to a control char
        self.assertEqual((meta["kind"], meta["host"]), ("other", "www.reddit.com"))  # hostile path: no metadata at all
        self.assertNotIn("subreddit", meta)
        meta = reddit_metadata("https://www.reddit.com/r/valid/comments/AB/")  # ids are lower-case base36
        self.assertEqual((meta["subreddit"], meta["kind"]), ("valid", "subreddit"))
        self.assertNotIn("post_id", meta)


if __name__ == "__main__":
    unittest.main()
