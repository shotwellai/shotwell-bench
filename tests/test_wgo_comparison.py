"""Focused checks for the WGO rank-comparison scoring arithmetic."""

import unittest

from scripts.score_wgo_comparison import eligible_edges, maximum_matches, snapped, subset_of, tiou


class WgoComparisonScoringTest(unittest.TestCase):
    def test_dataset_subset_names(self):
        self.assertEqual(subset_of('homer_1'), 'HomER')
        self.assertEqual(subset_of('robointer_1'), 'DROID')
        self.assertEqual(subset_of('galaxea_1'), 'Galaxea')

    def test_tiou_threshold(self):
        gold = [{'start_sec': 0, 'end_sec': 4}, {'start_sec': 4, 'end_sec': 8}]
        predictions = [{'start_sec': 0, 'end_sec': 4}, {'start_sec': 4.5, 'end_sec': 8}]
        self.assertEqual(tiou(predictions[0], gold[0]), 1)
        self.assertEqual(eligible_edges(predictions, gold), [(0, 0), (1, 1)])

    def test_outer_edges_are_snapped_without_mutating_output(self):
        original = [{'start_sec': .5, 'end_sec': 4}, {'start_sec': 4, 'end_sec': 7.5}]
        gold = [{'start_sec': 0, 'end_sec': 4}, {'start_sec': 4, 'end_sec': 8}]
        self.assertEqual(snapped(original, gold), gold)
        self.assertEqual(original[0]['start_sec'], .5)
        self.assertEqual(original[-1]['end_sec'], 7.5)

    def test_matching_is_one_to_one_and_maximum_cardinality(self):
        self.assertEqual(maximum_matches([(0, 0), (0, 1), (1, 0)], 2, 2), 2)
        self.assertEqual(maximum_matches([(0, 0), (1, 0)], 2, 1), 1)


if __name__ == '__main__':
    unittest.main()
