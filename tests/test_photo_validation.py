import unittest
import numpy as np

from scratch.photo_compare import average_precision, suggestion_thresholds
from scratch.photo_tags import LABELS


class PhotoValidationTests(unittest.TestCase):
    def test_tied_scores_do_not_receive_artificial_perfect_ranking(self):
        result = average_precision(np.array([[0.5], [0.5]]), np.array([[1], [0]]))
        self.assertAlmostEqual(result['mean_average_precision'], 0.5)

    def test_unsupported_labels_cannot_be_suggested(self):
        probabilities = np.full((10, len(LABELS)), 0.9)
        targets = np.zeros_like(probabilities)
        thresholds, evidence = suggestion_thresholds(probabilities, targets)
        self.assertTrue(all(threshold > 1 for threshold in thresholds))
        self.assertTrue(all(item is None for item in evidence))

    def test_validation_evidence_is_required_for_suggestions(self):
        probabilities = np.full((10, len(LABELS)), 0.1)
        targets = np.zeros_like(probabilities)
        probabilities[:6, 0] = 0.8
        targets[:6, 0] = 1
        thresholds, evidence = suggestion_thresholds(probabilities, targets)
        self.assertLessEqual(thresholds[0], 0.8)
        self.assertEqual(evidence[0]['validation_precision'], 1)
        self.assertEqual(evidence[0]['tp'], 6)


if __name__ == '__main__':
    unittest.main()
