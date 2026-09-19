import time
import unittest
import warnings
from datetime import datetime, timezone

from jose import jwt

from research_app.auth import auth
from research_app.db.models import User, _utc_now
from tests.base import ApiTestCase


class DatetimeTests(unittest.TestCase):
    def test_no_deprecation_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _utc_now()
            auth.create_access_token({"sub": "alice"})

    def test_utc_now_is_naive_utc(self):
        value = _utc_now()
        self.assertIsNone(value.tzinfo)  # matches the naive 'timestamp' columns
        expected = datetime.now(timezone.utc).replace(tzinfo=None)
        self.assertLess(abs((expected - value).total_seconds()), 5)

    def test_token_expiry_is_30_minutes_from_now(self):
        token = auth.create_access_token({"sub": "alice"})
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        self.assertLess(abs(payload["exp"] - (time.time() + 30 * 60)), 5)


class ModelDefaultTests(ApiTestCase):
    def test_created_at_is_set_per_row_not_at_import_time(self):
        # Regression: default=datetime.utcnow() was evaluated once at import, so
        # every row got the server-start timestamp.
        first = self.make_user("first")
        time.sleep(0.05)
        second = self.make_user("second")
        with self.SessionLocal() as db:
            a, b = db.get(User, first).created_at, db.get(User, second).created_at
        self.assertLess(a, b)
