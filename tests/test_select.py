import unittest

import numpy as np

from local_analysis import ensemble_select as E


class SelectTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n_ids, self.S = 60, 3
        self.N = self.n_ids * 4
        self.track_idx = np.tile(np.arange(4), self.n_ids)
        self.groups = np.repeat(np.arange(self.n_ids), 4)
        base = rng.uniform(0.6, 0.8, (self.N, self.S))
        base[:, 1] += 0.05                                      # system 1 is the best on average
        self.F = base

    def test_oracle_is_upper_bound(self):
        idx = E.choose_oracle(self.F)
        self.assertTrue((self.F[np.arange(self.N), idx] >= self.F.max(axis=1) - 1e-12).all())

    def test_mbr_picks_the_medoid(self):
        SIM = np.full((self.N, 3, 3), np.nan)
        for a, b, v in ((0, 1, 0.9), (0, 2, 0.5), (1, 2, 0.5)):  # 0 and 1 agree; 2 is the odd one out
            SIM[:, a, b] = SIM[:, b, a] = v
        chosen = E.choose_mbr(SIM)
        self.assertTrue(set(chosen) <= {0, 1})
        self.assertNotIn(2, set(chosen))

    def test_cv_folds_never_split_an_id(self):
        for tr, te in E.folds_by_group(self.groups):
            self.assertFalse(set(self.groups[tr]) & set(self.groups[te]))
            self.assertEqual(len(tr) + len(te), self.N)

    def test_per_track_best_cv_finds_best_system(self):
        chosen = E.cv_choose(self.F, self.groups, self.track_idx, method="per_track_best")
        self.assertGreater((chosen == 1).mean(), 0.9)

    def test_router_cv_learns_a_signal(self):
        # the first feature carries the F1 signal; a router must beat random choice out of sample
        rng = np.random.default_rng(1)
        X = rng.normal(size=(self.N, self.S, 3))
        F = 0.7 + 0.05 * X[:, :, 0]
        chosen = E.cv_choose(F, self.groups, self.track_idx, X=X, method="router")
        self.assertGreater(F[np.arange(self.N), chosen].mean(), F.mean() + 0.01)

    def test_mbr_weighted_cv_runs(self):
        SIM = np.random.default_rng(2).uniform(0.5, 0.9, (self.N, 3, 3))
        chosen = E.cv_choose(self.F, self.groups, self.track_idx, SIM=SIM, method="mbr_weighted")
        self.assertEqual(chosen.shape, (self.N,))

    def test_arabic_fraction(self):
        self.assertGreater(E.arabic_fraction("برج خليفة"), 0.9)
        self.assertEqual(E.arabic_fraction("Burj Khalifa"), 0.0)


if __name__ == "__main__":
    unittest.main()


class RouterRobustnessTests(unittest.TestCase):
    def test_all_nan_feature_column_is_ignored(self):
        rng = np.random.default_rng(3)
        n_ids, S = 30, 2
        N = n_ids * 4
        X = rng.normal(size=(N, S, 3))
        X[:, :, 1] = np.nan
        F = 0.7 + 0.05 * X[:, :, 0]
        groups = np.repeat(np.arange(n_ids), 4)
        chosen = E.cv_choose(F, groups, np.tile(np.arange(4), n_ids), X=X, method="router")
        self.assertEqual(chosen.shape, (N,))
