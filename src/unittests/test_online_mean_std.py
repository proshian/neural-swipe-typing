import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))


import unittest
from math import isclose

import torch

from data_obtaining_and_preprocessing.compute_trajectory_features_statistics import OnlineMeanStd


REL_TOL = 1e-5

class TestOnlineMeanStd(unittest.TestCase):
    def test_single_batch(self):
        x = torch.randn(1000)
        stats = OnlineMeanStd()
        stats.update(x)
        res = stats.get_statistics()

        self.assertTrue(isclose(res["mean"], x.mean().item(), rel_tol=REL_TOL))
        self.assertTrue(isclose(res["std"], x.std(unbiased=False).item(), rel_tol=REL_TOL))

    def test_multiple_batches(self):
        data = torch.randn(1000)
        stats = OnlineMeanStd()

        for i in range(0, 1000, 100):
            stats.update(data[i:i+100])

        res = stats.get_statistics()
        self.assertTrue(isclose(res["mean"], data.mean().item(), rel_tol=REL_TOL),
                        f'{res["mean"] = }, {data.mean().item() = }')
        self.assertTrue(isclose(res["std"], data.std(unbiased=False).item(), rel_tol=REL_TOL))

    def test_empty_input(self):
        stats = OnlineMeanStd()
        stats.update(torch.tensor([]))
        res = stats.get_statistics()

        self.assertEqual(res["mean"], 0.0)
        self.assertEqual(res["std"], 0.0)

    def test_single_value_updates(self):
        stats = OnlineMeanStd()
        data = torch.randn(1000)
        
        for el in data:
            stats.update(el)

        res = stats.get_statistics()
        self.assertTrue(isclose(res["mean"], data.mean().item(), rel_tol=REL_TOL))
        self.assertTrue(isclose(res["std"], data.std(unbiased=False).item(), rel_tol=REL_TOL))

    def test_consistency_across_batch_splits(self):
        data = torch.randn(500)
        stats1 = OnlineMeanStd()
        stats2 = OnlineMeanStd()

        stats1.update(data)
        for i in range(0, 500, 50):
            stats2.update(data[i:i+50])

        res1 = stats1.get_statistics()
        res2 = stats2.get_statistics()

        self.assertTrue(isclose(res1["mean"], res2["mean"], rel_tol=REL_TOL))
        self.assertTrue(isclose(res1["std"], res2["std"], rel_tol=REL_TOL))


if __name__ == "__main__":
    unittest.main()
