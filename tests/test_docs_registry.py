"""Official-docs registry: loading, validation, extension, and the URL verification that
decides whether a page may be labelled official documentation (a security boundary)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research_app.sources.official_docs import registry as reg
from research_app.sources.official_docs.registry import (
    DocsEntry,
    DocsRegistry,
    get_registry,
    load_entries,
    merge_entries,
    normalize_path,
    normalize_text,
    parse_entries,
)

EXPECTED_IDS = [
    "python", "fastapi", "uvicorn", "pydantic", "sqlalchemy", "alembic", "langgraph", "langchain",
    "qdrant", "postgresql", "docker", "docker-compose", "tavily", "openai-api", "claude-api", "groq",
]


def entry(id_, hosts=("example.com",), prefixes=(), aliases=None, **kw):
    return DocsEntry(
        id=id_, name=id_.title(), aliases=aliases or [id_],
        sites=[{"host": h, "path_prefixes": list(prefixes)} for h in hosts], **kw,
    )


class BuiltinRegistryTests(unittest.TestCase):
    def setUp(self):
        get_registry.cache_clear()
        self.addCleanup(get_registry.cache_clear)

    def test_builtin_file_loads_completely_and_quietly(self):
        # assertNoLogs: parse_entries warns for every skipped entry, so a typo in the
        # seed data (bad host, bad regex, unknown field) fails here.
        with self.assertNoLogs("research_app.sources.official_docs.registry", level="WARNING"):
            entries = load_entries(reg.BUILTIN_REGISTRY_PATH)
        self.assertEqual([e.id for e in entries], EXPECTED_IDS)

    def test_every_site_is_https_hostname_only_and_prefixes_are_plain_paths(self):
        for e in get_registry().entries:
            for site in e.sites:
                self.assertRegex(site.host, r"^[a-z0-9.-]+\.[a-z]+$", e.id)
                for p in site.path_prefixes:
                    self.assertTrue(p.startswith("/") and not p.endswith("/"), (e.id, p))

    def test_seed_registry_shape(self):
        r = get_registry()
        self.assertEqual(r.get("fastapi").sites[0].host, "fastapi.tiangolo.com")
        # Shared official hosts are protected by path prefixes.
        self.assertEqual(r.get("langgraph").sites[0].path_prefixes, ("/oss/python/langgraph",))
        self.assertEqual(r.get("qdrant").sites[0].path_prefixes, ("/documentation",))
        self.assertEqual(r.get("docker-compose").sites[0].host, "docs.docker.com")
        self.assertIsNone(r.get("nope"))


class MatchUrlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_registry.cache_clear()
        cls.r = get_registry()

    def match(self, url, **kw):
        return self.r.match_url(url, **kw)

    def test_accepts_official_pages_with_technology_prefix_and_version(self):
        cases = [
            ("https://fastapi.tiangolo.com/tutorial/first-steps/", "fastapi", None, None),
            ("https://www.postgresql.org/docs/18/sql-createindex.html", "postgresql", "/docs", "18"),
            ("https://www.postgresql.org/docs/current/sql-createindex.html", "postgresql", "/docs", None),
            ("https://www.postgresql.org/docs/9.6/index.html", "postgresql", "/docs", "9.6"),
            ("https://docs.python.org/3.12/library/asyncio.html", "python", None, "3.12"),
            ("https://docs.python.org/3/library/asyncio.html", "python", None, "3"),
            ("https://docs.sqlalchemy.org/en/20/orm/", "sqlalchemy", None, "20"),
            ("https://pydantic.dev/docs/validation/2.11/concepts/models/", "pydantic", "/docs/validation", "2.11"),
            ("https://pydantic.dev/docs/validation/latest/get-started/", "pydantic", "/docs/validation", None),
            ("https://docs.langchain.com/oss/python/langgraph/overview", "langgraph", "/oss/python/langgraph", None),
            ("https://docs.langchain.com/oss/python/langchain/overview", "langchain", "/oss/python/langchain", None),
            ("https://qdrant.tech/documentation/guides/installation/", "qdrant", "/documentation", None),
            ("https://uvicorn.dev/deployment/", "uvicorn", None, None),
            ("https://developers.openai.com/api/docs/quickstart", "openai-api", "/api", None),
            ("https://platform.claude.com/docs/en/intro", "claude-api", "/docs", None),
            ("https://console.groq.com/docs/quickstart", "groq", "/docs", None),
        ]
        for url, tech, prefix, version in cases:
            with self.subTest(url=url):
                m = self.match(url)
                self.assertIsNotNone(m)
                self.assertEqual((m.entry.id, m.matched_prefix, m.version), (tech, prefix, version))

    def test_shared_host_resolves_to_the_most_specific_prefix(self):
        docker_host = "https://docs.docker.com"
        cases = {
            "/compose/how-tos/": "docker-compose",
            "/reference/compose-file/services/": "docker-compose",
            "/reference/cli/docker/compose/up/": "docker-compose",
            "/engine/install/": "docker",
            "/reference/cli/docker/run/": "docker",
        }
        for path, tech in cases.items():
            with self.subTest(path=path):
                self.assertEqual(self.match(docker_host + path).entry.id, tech)

    def test_www_is_ignored_on_both_sides(self):
        self.assertEqual(self.match("https://www.fastapi.tiangolo.com/").entry.id, "fastapi")
        self.assertEqual(self.match("https://postgresql.org/docs/18/").entry.id, "postgresql")

    def test_look_alike_and_malicious_hosts_are_rejected(self):
        for url in (
            "https://fastapi.tiangolo.com.evil.com/tutorial/",
            "https://evil.com/fastapi.tiangolo.com/",
            "https://evilfastapi.tiangolo.com/",
            "https://fastapi-tiangolo.com/",
            "https://fastapi.tiangolo.com@evil.com/",
            "https://evil.com@fastapi.tiangolo.com/",  # credentials: never accepted
            "https://user:pw@fastapi.tiangolo.com/",
            "https://docs.fastapi.tiangolo.com/",  # subdomains are not implied
            "https://x.fastapi.tiangolo.com/",
            "https://fastapi.tiangolo.co/",
            "https://fastapi.tiangolo.com.:8443/",
            "https://faѕtapi.tiangolo.com/",  # Cyrillic 's'
            "https://fastapi.tiangolo.cοm/",  # Greek omicron
            "https://xn--fastapi-tiangolo-x.com/",
            "https://127.0.0.1/",
            "https://[::1]/",
            "https://wiki.postgresql.org/docs/18/",
            "https://evil.example/docs/18/",
        ):
            with self.subTest(url=url):
                self.assertIsNone(self.match(url))

    def test_scheme_and_port_are_enforced(self):
        for url in (
            "http://fastapi.tiangolo.com/",
            "ftp://fastapi.tiangolo.com/",
            "javascript:alert(1)",
            "//fastapi.tiangolo.com/",
            "fastapi.tiangolo.com/",
            "https://fastapi.tiangolo.com:8443/",
            "https://fastapi.tiangolo.com:80/",
        ):
            with self.subTest(url=url):
                self.assertIsNone(self.match(url))
        self.assertIsNotNone(self.match("https://fastapi.tiangolo.com:443/"))

    def test_path_prefix_is_enforced_on_shared_hosts(self):
        for url in (
            "https://docs.langchain.com/langsmith/home",
            "https://docs.langchain.com/oss/javascript/langgraph/overview",
            "https://docs.langchain.com/oss/python/langgraphevil/x",  # not a segment boundary
            "https://docs.langchain.com/",
            "https://qdrant.tech/blog/",
            "https://qdrant.tech/documentationx/",
            "https://www.postgresql.org/about/news/",
            "https://pydantic.dev/blog/",
            "https://console.groq.com/keys",
        ):
            with self.subTest(url=url):
                self.assertIsNone(self.match(url))
        self.assertIsNotNone(self.match("https://docs.langchain.com/oss/python/langgraph"))  # the prefix itself

    def test_dot_segments_and_encoding_cannot_smuggle_a_path_into_a_prefix(self):
        # Resolved the way a client would: this is /engine/, not /compose/.
        self.assertEqual(self.match("https://docs.docker.com/compose/../engine/").entry.id, "docker")
        self.assertEqual(self.match("https://docs.docker.com/compose/%2e%2e/engine/").entry.id, "docker")
        self.assertIsNone(self.match("https://qdrant.tech/documentation/../blog/"))
        self.assertIsNone(self.match("https://qdrant.tech/documentation/%2e%2e/blog/"))
        self.assertIsNone(self.match("https://qdrant.tech/documentation\\..\\blog"))
        self.assertIsNone(self.match("https://qdrant.tech/documentation/\x00"))
        self.assertIsNone(self.match("https://qdrant.tech/" + "a" * 3000))

    def test_allowed_hosts_narrow_the_candidates(self):
        url = "https://docs.python.org/3/library/os.html"
        self.assertIsNotNone(self.match(url))
        self.assertIsNone(self.match(url, allowed_hosts=["fastapi.tiangolo.com"]))
        self.assertIsNotNone(self.match(url, allowed_hosts=["DOCS.python.org."]))
        self.assertIsNone(self.match(url, allowed_hosts=[]))

    def test_garbage_input_never_raises(self):
        for value in ("", "   ", "not a url", "https://", "https://[bad", "https://exa mple.com/", None, 42):
            with self.subTest(value=value):
                self.assertIsNone(self.match(value))

    def test_matching_is_deterministic_regardless_of_call_order(self):
        urls = ["https://docs.docker.com/compose/", "https://docs.docker.com/engine/", "https://docs.python.org/3/"]
        first = [self.match(u).entry.id for u in urls]
        second = [self.match(u).entry.id for u in reversed(urls)][::-1]
        self.assertEqual(first, second)

    def test_equal_specificity_falls_back_to_registry_order(self):
        a, b = entry("first", hosts=["shared.example"]), entry("second", hosts=["shared.example"])
        self.assertEqual(DocsRegistry([a, b]).match_url("https://shared.example/x").entry.id, "first")
        self.assertEqual(DocsRegistry([b, a]).match_url("https://shared.example/x").entry.id, "second")

    def test_alias_patterns_have_a_fixed_order_and_first_owner_wins(self):
        a = entry("one", aliases=["docker", "docker compose"])
        b = entry("two", aliases=["docker", "kompose"])
        with self.assertLogs("research_app.sources.official_docs.registry", level="WARNING"):
            r = DocsRegistry([a, b])
        aliases = [p.alias for p in r.alias_patterns()]
        self.assertEqual(aliases, sorted(aliases, key=lambda x: (-len(x), x)))
        owners = {p.alias: p.entry.id for p in r.alias_patterns()}
        self.assertEqual(owners["docker"], "one")
        self.assertEqual(owners["kompose"], "two")


class HelperTests(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Docker-Compose_File  "), "docker compose file")
        self.assertEqual(normalize_text("ＦａｓｔＡＰＩ"), "fastapi")  # NFKC

    def test_normalize_path(self):
        self.assertEqual(normalize_path("/a/b/../c/./d//e"), "/a/c/d/e")
        self.assertEqual(normalize_path(""), "/")
        self.assertEqual(normalize_path("/../../x"), "/x")
        self.assertEqual(normalize_path("/a%2Fb"), "/a/b")
        self.assertIsNone(normalize_path("/a\\b"))
        self.assertIsNone(normalize_path("/a\x07b"))


class ParseAndMergeTests(unittest.TestCase):
    GOOD = {"id": "good", "name": "Good", "aliases": ["good"], "sites": [{"host": "docs.good.dev"}]}

    def test_invalid_entries_are_skipped_and_logged_without_stopping_the_rest(self):
        bad = [
            {**self.GOOD, "id": "url-host", "sites": [{"host": "https://docs.good.dev"}]},
            {**self.GOOD, "id": "wild", "sites": [{"host": "*.good.dev"}]},
            {**self.GOOD, "id": "port", "sites": [{"host": "docs.good.dev:8443"}]},
            {**self.GOOD, "id": "prefix", "sites": [{"host": "docs.good.dev", "path_prefixes": ["docs"]}]},
            {**self.GOOD, "id": "dots", "sites": [{"host": "docs.good.dev", "path_prefixes": ["/a/../b"]}]},
            {**self.GOOD, "id": "regex", "sites": [{"host": "docs.good.dev", "version_patterns": ["("]}]},
            {**self.GOOD, "id": "nogroup", "sites": [{"host": "docs.good.dev", "version_patterns": ["\\d+"]}]},
            {**self.GOOD, "id": "Bad Id"},
            {**self.GOOD, "id": "extra", "surprise": True},
            {**self.GOOD, "id": "noalias", "aliases": []},
            {**self.GOOD, "id": "nosites", "sites": []},
            {**self.GOOD, "id": "shortalias", "aliases": ["x"]},
            "not a dict", 7, None,
        ]
        with self.assertLogs("research_app.sources.official_docs.registry", level="WARNING") as logs:
            entries = parse_entries({"technologies": bad + [self.GOOD]}, source="test.json")
        self.assertEqual([e.id for e in entries], ["good"])
        self.assertEqual(len(logs.records), len(bad))
        self.assertNotIn("evil", "\n".join(logs.output))

    def test_top_level_shape_errors_raise(self):
        for data in ([], {}, {"technologies": "x"}, None):
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_entries(data, source="t")

    def test_duplicate_ids_keep_the_first(self):
        other = {**self.GOOD, "name": "Second"}
        with self.assertLogs("research_app.sources.official_docs.registry", level="WARNING"):
            entries = parse_entries({"technologies": [self.GOOD, other]}, source="t")
        self.assertEqual([e.name for e in entries], ["Good"])

    def test_root_prefix_means_the_whole_host_and_prefixes_are_normalised(self):
        e = DocsEntry.model_validate({**self.GOOD, "sites": [{"host": "Docs.Good.dev", "path_prefixes": ["/"]}]})
        self.assertEqual((e.sites[0].host, e.sites[0].path_prefixes), ("docs.good.dev", ()))
        e = DocsEntry.model_validate({**self.GOOD, "sites": [{"host": "docs.good.dev", "path_prefixes": ["/a/", "/a"]}]})
        self.assertEqual(e.sites[0].path_prefixes, ("/a",))

    def test_aliases_are_normalised(self):
        e = DocsEntry.model_validate({**self.GOOD, "aliases": ["Good-Tool", "good tool"]})
        self.assertEqual(e.aliases, ("good tool",))

    def test_merge_replaces_by_id_in_place_and_appends_new(self):
        a, b, c = entry("aa"), entry("bb"), entry("cc")
        b2 = DocsEntry(id="bb", name="B2", aliases=["bb"], sites=[{"host": "b.example.org"}])
        merged = merge_entries([a, b, c], [entry("zz"), b2])
        self.assertEqual([e.id for e in merged], ["aa", "bb", "cc", "zz"])
        self.assertEqual(merged[1].name, "B2")


class GetRegistryTests(unittest.TestCase):
    def setUp(self):
        get_registry.cache_clear()
        self.addCleanup(get_registry.cache_clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, content):
        path = Path(self.tmp.name) / name
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return str(path)

    def test_user_registry_extends_and_overrides_the_builtin_one(self):
        extra = self.write("extra.json", {"technologies": [
            {"id": "acme", "name": "Acme", "aliases": ["acme cloud"], "sites": [{"host": "docs.acme.example"}]},
            {"id": "fastapi", "name": "FastAPI (mirror)", "aliases": ["fastapi"],
             "sites": [{"host": "fastapi.example.org"}]},
        ]})
        r = get_registry(extra)
        self.assertIsNotNone(r.get("acme"))
        self.assertEqual(r.get("fastapi").name, "FastAPI (mirror)")
        self.assertEqual(len(r), len(EXPECTED_IDS) + 1)
        self.assertIsNone(r.match_url("https://fastapi.tiangolo.com/"))  # overridden
        self.assertEqual(r.match_url("https://docs.acme.example/x").entry.id, "acme")
        # The built-in registry is unaffected.
        self.assertEqual(get_registry().get("fastapi").name, "FastAPI")

    def test_broken_user_registry_is_ignored(self):
        for name, content in (("bad.json", "{not json"), ("shape.json", "[]")):
            with self.subTest(file=name), self.assertLogs("research_app.sources.official_docs.registry", level="ERROR"):
                self.assertEqual(len(get_registry(self.write(name, content))), len(EXPECTED_IDS))
        with self.assertLogs("research_app.sources.official_docs.registry", level="ERROR"):
            self.assertEqual(len(get_registry(str(Path(self.tmp.name) / "missing.json"))), len(EXPECTED_IDS))

    def test_user_registry_with_some_bad_entries_keeps_the_good_ones(self):
        extra = self.write("mixed.json", {"technologies": [
            {"id": "bad", "name": "Bad", "aliases": ["bad"], "sites": [{"host": "not a host"}]},
            {"id": "acme", "name": "Acme", "aliases": ["acme cloud"], "sites": [{"host": "docs.acme.example"}]},
        ]})
        with self.assertLogs("research_app.sources.official_docs.registry", level="WARNING"):
            r = get_registry(extra)
        self.assertIsNone(r.get("bad"))
        self.assertIsNotNone(r.get("acme"))

    def test_unusable_builtin_file_yields_an_empty_registry_not_an_exception(self):
        with mock.patch.object(reg, "BUILTIN_REGISTRY_PATH", Path(self.tmp.name) / "nope.json"):
            with self.assertLogs("research_app.sources.official_docs.registry", level="ERROR"):
                r = get_registry()
        self.assertEqual(len(r), 0)
        self.assertIsNone(r.match_url("https://fastapi.tiangolo.com/"))

    def test_registry_is_cached_per_path(self):
        self.assertIs(get_registry(), get_registry())


if __name__ == "__main__":
    unittest.main()
