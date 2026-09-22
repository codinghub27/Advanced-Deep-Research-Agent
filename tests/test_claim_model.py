"""P1.6 -- Claim/evidence domain model. Pure model tests, no LLM/network/DB."""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from research_app.domain import (
    Citation,
    Claim,
    ClaimSupportStatus,
    ClaimType,
    ClaimVerificationStatus,
    ResearchQuery,
    ResearchRun,
    SourceFreshnessStatus,
)


class ClaimDefaultsTests(unittest.TestCase):
    def test_minimal_claim_has_safe_defaults(self):
        claim = Claim(text="FastAPI is built on Starlette.")
        self.assertEqual(claim.claim_type, ClaimType.FACT)
        # A fresh claim is unsupported and unverified until evidence/verification say otherwise
        # -- nothing is trusted by default.
        self.assertEqual(claim.support_status, ClaimSupportStatus.UNSUPPORTED)
        self.assertEqual(claim.verification_status, ClaimVerificationStatus.UNVERIFIED)
        self.assertFalse(claim.conflict_status)
        self.assertEqual(claim.evidence_ids, [])
        self.assertEqual(claim.source_ids, [])
        self.assertIsNotNone(claim.claim_id)

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValidationError):
            Claim(text="")

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            Claim(text="x", not_a_real_field=1)  # type: ignore[call-arg]

    def test_evidence_strength_must_be_in_0_1(self):
        with self.assertRaises(ValidationError):
            Claim(text="x", evidence_strength=1.5)
        Claim(text="x", evidence_strength=0.7)  # does not raise

    def test_opinion_claim_can_be_not_applicable(self):
        claim = Claim(text="I think FastAPI is the best framework.", claim_type=ClaimType.OPINION,
                       support_status=ClaimSupportStatus.NOT_APPLICABLE)
        self.assertEqual(claim.support_status, ClaimSupportStatus.NOT_APPLICABLE)

    def test_all_claim_types_are_constructible(self):
        for claim_type in ClaimType:
            with self.subTest(claim_type=claim_type):
                Claim(text="x", claim_type=claim_type)

    def test_all_support_statuses_are_constructible(self):
        for status in ClaimSupportStatus:
            with self.subTest(status=status):
                Claim(text="x", support_status=status)

    def test_freshness_status_is_optional_and_typed(self):
        claim = Claim(text="x", freshness_status=SourceFreshnessStatus.STALE)
        self.assertEqual(claim.freshness_status, SourceFreshnessStatus.STALE)
        with self.assertRaises(ValidationError):
            Claim(text="x", freshness_status="not_a_real_status")  # type: ignore[arg-type]

    def test_run_and_citation_linkage_fields_round_trip(self):
        claim = Claim(text="x", run_id="run-1", citation_id="cit-1",
                       evidence_ids=["e1", "e2"], source_ids=["s1"], excerpts=["quoted text"])
        self.assertEqual(claim.run_id, "run-1")
        self.assertEqual(claim.citation_id, "cit-1")
        self.assertEqual(claim.evidence_ids, ["e1", "e2"])
        self.assertEqual(claim.source_ids, ["s1"])
        self.assertEqual(claim.excerpts, ["quoted text"])


class CitationIdTests(unittest.TestCase):
    def test_citation_gets_a_stable_id_distinct_from_marker(self):
        citation = Citation(marker="[1]", source_id="abc", url="https://example.com/a")
        self.assertIsNotNone(citation.citation_id)
        self.assertNotEqual(citation.citation_id, citation.marker)

    def test_two_citations_get_different_ids(self):
        a = Citation(marker="[1]", source_id="abc", url="https://example.com/a")
        b = Citation(marker="[1]", source_id="abc", url="https://example.com/a")
        self.assertNotEqual(a.citation_id, b.citation_id)


class ResearchRunClaimsTests(unittest.TestCase):
    def test_research_run_carries_a_claims_list_defaulting_to_empty(self):
        run = ResearchRun(query=ResearchQuery(text="q"))
        self.assertEqual(run.claims, [])

    def test_research_run_accepts_claims(self):
        run = ResearchRun(query=ResearchQuery(text="q"), claims=[Claim(text="x")])
        self.assertEqual(len(run.claims), 1)


if __name__ == "__main__":
    unittest.main()
