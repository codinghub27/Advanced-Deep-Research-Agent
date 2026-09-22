"""Server terminal noise: per-node / per-search INFO lines and /health access lines are hidden
unless LOG_NODE_TRACE=true (or LOG_LEVEL=DEBUG). Warnings and per-request lines stay."""
import logging
import os
import unittest
from unittest import mock

from research_app.logging_config import NODE_TRACE_LOGGERS, configure_logging

WATCHED = ("research_app.agent", "research_app.agent.state", "research_app.sources.routing.router",
           "research_app.main", "uvicorn.access")


class LoggingConfigTests(unittest.TestCase):
    def setUp(self):
        saved = {n: (logging.getLogger(n).level, list(logging.getLogger(n).filters)) for n in WATCHED}
        root_level = logging.getLogger().level

        def restore():
            logging.getLogger().setLevel(root_level)
            for n, (lvl, filters) in saved.items():
                logging.getLogger(n).setLevel(lvl)
                logging.getLogger(n).filters = filters
        self.addCleanup(restore)

    def configure(self, **env):
        clean = {k: v for k, v in os.environ.items() if k not in ("LOG_LEVEL", "LOG_NODE_TRACE")}
        with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
            configure_logging()
        # In a real server basicConfig sets the root level; the test runner's root already has
        # handlers, which makes basicConfig a no-op, so do what a fresh process would have done.
        logging.getLogger().setLevel(getattr(logging, env.get("LOG_LEVEL", "INFO")))

    def enabled(self, name, level=logging.INFO):
        return logging.getLogger(name).isEnabledFor(level)

    def test_node_and_routing_info_lines_are_hidden_by_default(self):
        self.configure()
        self.assertFalse(self.enabled("research_app.agent.state"))
        self.assertFalse(self.enabled("research_app.sources.routing.router"))

    def test_warnings_and_errors_from_them_still_show(self):
        self.configure()
        self.assertTrue(self.enabled("research_app.agent.state", logging.WARNING))
        self.assertTrue(self.enabled("research_app.sources.routing.router", logging.ERROR))

    def test_the_per_request_line_stays(self):
        self.configure()
        self.assertTrue(self.enabled("research_app.main"))

    def test_trace_mode_brings_everything_back(self):
        self.configure(LOG_NODE_TRACE="true")
        self.assertTrue(self.enabled("research_app.agent.state"))
        self.assertTrue(self.enabled("research_app.sources.routing.router"))

    def test_debug_level_implies_trace(self):
        self.configure(LOG_LEVEL="DEBUG")
        self.assertTrue(self.enabled("research_app.agent.state", logging.DEBUG))

    def test_an_error_only_level_is_respected(self):
        self.configure(LOG_LEVEL="ERROR")
        self.assertFalse(self.enabled("research_app.agent.state", logging.WARNING))

    def record(self, message):
        return logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, message, None, None)

    def test_health_check_access_lines_are_dropped_but_other_requests_are_kept(self):
        self.configure()
        access = logging.getLogger("uvicorn.access")
        self.assertFalse(access.filter(self.record('127.0.0.1:1 - "GET /health HTTP/1.1" 200 OK')))
        self.assertTrue(access.filter(self.record('127.0.0.1:1 - "POST /api/research/stream HTTP/1.1" 200 OK')))
        self.assertTrue(access.filter(self.record('127.0.0.1:1 - "GET /healthy-recipes HTTP/1.1" 200 OK')))

    def test_calling_it_twice_adds_one_filter_and_trace_removes_it(self):
        self.configure()
        self.configure()
        self.assertEqual(len(logging.getLogger("uvicorn.access").filters), 1)
        self.configure(LOG_NODE_TRACE="true")
        self.assertEqual(logging.getLogger("uvicorn.access").filters, [])

    def test_the_quiet_list_covers_the_modules_in_the_terminal_output(self):
        for name in ("research_app.agent.state", "research_app.sources.routing.router",
                     "research_app.sources.routing.task_router"):
            self.assertTrue(any(name == p or name.startswith(p + ".") for p in NODE_TRACE_LOGGERS), name)


if __name__ == "__main__":
    unittest.main()
