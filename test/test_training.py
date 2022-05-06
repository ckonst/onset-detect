# -*- coding: utf-8 -*-

import numpy as np
import unittest
import train
import torch

from file_reading.file_reading import file_to_ndarray
from onset_detect import get_lmfs

class TrainingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fs, cls.test_audio, _ = file_to_ndarray('../baseline/pop_shuffle.wav')
        cls.test_onsets = np.load('../baseline/onsets.npy')
        cls.clicks = file_to_ndarray('../baseline/click_track.wav')
        fs = cls.fs
        cls.test_spectrogram = get_lmfs(fs, cls.test_audio, 1024, int(0.001 * fs), 20.0, 0.5 * fs)

    def test_evaluate_batch(self):
        x = torch.tensor([[0,0,0,0,0,0,0,0]])
        y = torch.tensor([[0,0,0,0,0,0,0,0]])
        assert train.evaluate_batch(x, y, 0.01, 100, 1)\
            == (0.0, 0.0, 0.0)
        x = torch.tensor([[1,0,1,0,1,0,1,0]])
        y = torch.tensor([[1,0,0,1,1,0,1,1]])
        assert train.evaluate_batch(x, y, 0.01, 100, 1)\
            == (1.0, 1.0, 1.0)

    def test_evaluate_frame(self):
        x = torch.tensor([0,0,0,0,0,0,0,0])
        y = torch.tensor([0,0,0,0,0,0,0,0])
        assert train.evaluate_frame(x, y, 0.01, 100, 1)\
            == (0.0, 0.0, 0.0)
        x = torch.tensor([1,0,1,0,1,0,1,0])
        y = torch.tensor([1,0,0,1,1,0,1,1])
        assert train.evaluate_frame(x, y, 0.01, 100, 1)\
            == (1.0, 1.0, 1.0)

    def test_evaluate_frame_naive(self):
        x = torch.tensor([0,0,0,0,0,0,0,0])
        y = torch.tensor([0,0,0,0,0,0,0,0])
        assert train.evaluate_frame_naive(x, y)\
            == (0.0, 0.0, 0.0)
        x = torch.tensor([1,0,1,0,1,0,1,0])
        y = torch.tensor([1,0,0,1,1,0,1,1])
        assert train.evaluate_frame_naive(x, y)\
            == (1.0, 1.0, 1.0)

    def test_fscore_precision_recall(self):
        tp, fp, fn = 100, 300, 400
        assert train.fscore_precision_recall(tp, fp, fn)\
            == (0.2222222222222222, 0.25, 0.2)
        tp, fp, fn = 0, 0, 0
        assert train.fscore_precision_recall(tp, fp, fn)\
            == (0.0, 0.0, 0.0)

if __name__ == '__main__':
    unittest.main()
