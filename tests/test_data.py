import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import numpy as np
from bank_marketing.data import (synthetic_frame, validate_frame, split_indices, build_preprocessor,
                                 transform_split, NUMERIC)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame(500)
        self.x, self.y, self.meta = validate_frame(self.frame)
        self.idx = split_indices(self.y)

    def test_target_encoding(self):
        np.testing.assert_array_equal(self.y, (self.frame.y == 'yes').astype(int))

    def test_stratified_disjoint_split(self):
        self.assertEqual([len(v) for v in self.idx.values()], [320, 80, 100])
        self.assertEqual(len(set(np.concatenate(list(self.idx.values())))), 500)
        for idx in self.idx.values():
            self.assertAlmostEqual(self.y[idx].mean(), .2)
        for key in self.idx:
            np.testing.assert_array_equal(self.idx[key], split_indices(self.y)[key])

    def test_scaler_uses_only_train(self):
        self.x.loc[self.idx['test'], 'age'] = 100000
        prep = build_preprocessor(self.x.iloc[self.idx['train']])
        num = [c for c in NUMERIC if c != 'duration']
        np.testing.assert_allclose(prep.named_transformers_['numeric'].mean_,
                                   self.x.iloc[self.idx['train']][num].mean())
        self.assertNotAlmostEqual(prep.named_transformers_['numeric'].mean_[0], self.x.age.mean())

    def test_unknown_and_unseen_categories(self):
        prep = build_preprocessor(self.x.iloc[self.idx['train']])
        self.assertIn('unknown', prep.named_transformers_['categorical'].categories_[0])
        self.x.loc[self.idx['validation'], 'job'] = 'unseen_validation'
        data = transform_split(prep, self.x, self.y, self.idx['validation'])
        self.assertTrue(np.isfinite(data.x).all())
        self.assertNotIn('unseen_validation', prep.named_transformers_['categorical'].categories_[0])

    def test_feature_modes(self):
        primary = build_preprocessor(self.x.iloc[self.idx['train']])
        secondary = build_preprocessor(self.x.iloc[self.idx['train']], 'duration_inclusive')
        self.assertNotIn('numeric__duration', primary.get_feature_names_out())
        self.assertIn('numeric__day', primary.get_feature_names_out())
        self.assertNotIn('numeric__day_of_week', primary.get_feature_names_out())
        with self.assertRaisesRegex(ValueError, 'Schema'):
            validate_frame(self.frame.rename(columns={'day': 'day_of_week'}))
        self.assertIn('numeric__duration', secondary.get_feature_names_out())
        for col in ['pdays', 'previous']:
            self.assertIn('numeric__' + col, primary.get_feature_names_out())
        self.assertEqual(len(secondary.get_feature_names_out()), len(primary.get_feature_names_out()) + 1)

    def test_actual_nan_rejected(self):
        self.frame.loc[0, 'job'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Actual NaN'):
            validate_frame(self.frame)

    def test_invalid_target_and_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Schema'):
            validate_frame(self.frame.drop(columns='age'))
        self.frame.loc[0, 'y'] = 'maybe'
        with self.assertRaisesRegex(ValueError, 'Target'):
            validate_frame(self.frame)

    def test_numeric_inf_rejected(self):
        self.frame.loc[0, 'balance'] = np.inf
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            validate_frame(self.frame)

