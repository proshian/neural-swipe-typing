import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))


import unittest

import torch

from feature_extraction.distance_getter import compute_pairwise_distances


class TestDistance(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_distance_shape_1(self):
        N_KEYS = 30
        WIDTH = MAX_COORD = 1080
        HEIGHT = 667
        dots = torch.stack(torch.meshgrid(torch.arange(WIDTH), torch.arange(HEIGHT), indexing='xy'), dim=-1)  # (WIDTH, HEIGHT, 2)
        centers = torch.randint(0, MAX_COORD, (N_KEYS, 2))
        result = compute_pairwise_distances(dots, centers)
        self.assertEqual(result.shape, (WIDTH, HEIGHT, N_KEYS))

    def test_distance_shape_2(self):
        N_KEYS = 30
        MAX_COORD = 1080
        DOT_DIMS = (100, 50, 90)
        dots = torch.randint(0, MAX_COORD, (*DOT_DIMS, 2))
        centers = torch.randint(0, MAX_COORD, (N_KEYS, 2))
        result = compute_pairwise_distances(dots, centers)
        self.assertEqual(result.shape, (*DOT_DIMS, N_KEYS))

    def testcase_dots_1(self):
        dots = torch.tensor([[1, 2], [3, 4], [5, 6]])
        centers = torch.tensor([[1, 2], [3, 4]])
        result = compute_pairwise_distances(dots, centers)
        expected = torch.tensor([[0, 8], [8, 0], [32, 8]])
        self.assertTrue(torch.allclose(result, expected))

    def testcase_grid_1(self):
        WIDTH = 2
        HEIGHT = 3
        dots = torch.stack(torch.meshgrid(torch.arange(WIDTH), torch.arange(HEIGHT), indexing='xy'), dim=-1)
        centers = torch.tensor([[4, 6]])
        result = compute_pairwise_distances(dots, centers)
        expected = torch.tensor([[[52], [41], [32]], [[45], [34], [25]]])
        self.assertTrue(torch.allclose(result, expected))


if __name__ == '__main__':
    unittest.main()
