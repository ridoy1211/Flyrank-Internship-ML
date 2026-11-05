"""Parity checks for the canonical research package against Fix05–07 archives."""

from datetime import date
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = REPOSITORY_ROOT / "work"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import chronological_fix05 as archived_evaluation
import count_noise_fix06 as archived_count_noise
import paired_queue_fix07 as archived_paired
from test_chronological_fix05 import synthetic_aggregate
from work.research import count_noise, evaluation, paired


ARCHIVED_FIX05_PROTOCOL_SHA256 = (
    "fa08ab634857e99ecdd9b194c93105795258b31ddaed49454c55f2d8a478e0a8"
)


class ResearchPackageTests(unittest.TestCase):
    def test_protocol_hash_matches_archived_fix05_receipt(self):
        self.assertEqual(evaluation.protocol_hash05(), ARCHIVED_FIX05_PROTOCOL_SHA256)

    def test_chronological_return_parity_and_optional_callbacks(self):
        episodes = {}
        audits = []
        for origin in archived_evaluation.EPISODES05:
            episodes[origin], audit = archived_evaluation.build_episode05(
                synthetic_aggregate(origin), origin
            )
            audits.append(audit)

        def deterministic_scores(train, test):
            base = test.ctr_prior28.to_numpy(dtype=float)
            scores = {
                name: base + offset / 100.0
                for offset, name in enumerate(evaluation.METHODS05)
            }
            return scores, {"n_training_rows": len(train)}

        with (
            patch.object(archived_evaluation, "score_models05", side_effect=deterministic_scores),
            patch.object(evaluation, "score_models05", side_effect=deterministic_scores),
        ):
            archived_result = archived_evaluation.run_chronological05(episodes, audits)

            progress_events = []
            sink_events = []

            def score_sink(origin, arm, test_frame, scores):
                sink_events.append((origin, arm, test_frame.copy(), scores.copy()))

            canonical_result = evaluation.run_chronological05(
                episodes,
                audits,
                score_sink=score_sink,
                progress=progress_events.append,
            )

        self.assertEqual(canonical_result, archived_result)
        self.assertEqual(len(progress_events), len(sink_events))
        self.assertTrue(progress_events)
        scored_arms = [
            (fold["origin"], arm["arm"])
            for fold in canonical_result["folds"]
            for arm in fold["arms"]
            if arm.get("status") == "scored"
        ]
        self.assertEqual([(origin, arm) for origin, arm, _, _ in sink_events], scored_arms)
        for origin, arm, test_frame, scores in sink_events:
            self.assertIsInstance(origin, str)
            self.assertIn(arm, {"known_clients", "naturally_unseen_clients", "hash_heldout_clients"})
            self.assertEqual(len(test_frame), len(scores["Logistic Regression"]))
            self.assertEqual(set(scores), set(evaluation.METHODS05))
        self.assertFalse(hasattr(evaluation, "aggregation_sql05"))

    def test_count_noise_probability_matches_archived_helper(self):
        args = (
            np.array([100, 100, 150, 100, 250]),
            np.array([1, 20, 150, 17, 5]),
            np.array([100, 90, 150, 111, 250]),
            np.array([0, 17, 50, 2, 3]),
        )
        expected = archived_count_noise.conditional_signature_probability06(*args)
        actual = count_noise.conditional_signature_probability06(*args)
        np.testing.assert_array_equal(actual, expected)

    def test_paired_queue_bounds_match_archived_helper(self):
        frame = pd.DataFrame(
            {
                "decoupling_signature": pd.array([1, pd.NA, 0, 1, pd.NA], dtype="Int64"),
                "client_hash_id": ["a", "a", "b", "b", "c"],
            }
        )
        scores_a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        scores_b = np.array([1.0, 4.0, 3.0, 2.0, 5.0])

        self.assertEqual(
            paired.paired_bounds07(frame, scores_a, scores_b, k=3),
            archived_paired.paired_bounds07(frame, scores_a, scores_b, k=3),
        )

        aliases = {"a": "client_001", "b": "client_002", "c": "client_003"}
        self.assertEqual(
            paired.client_deletion07(frame, scores_a, scores_b, aliases, k=2),
            archived_paired.client_deletion07(frame, scores_a, scores_b, aliases, k=2),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
