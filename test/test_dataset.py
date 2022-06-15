# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:57:44 2022

@author: Christian Konstantinov
"""

import unittest

from dataset.dataset import OnsetDataset
from model.hyperparameters import DSP, ML

from torch.utils.data import Subset

class TrainingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ml = ML()
        cls.dsp = DSP()
        cls.ods = OnsetDataset(ml=cls.ml, dsp=cls.dsp)

    def test_split_train_test_dev(self):
        train, test, dev = self.ods.split_train_test_dev()
        assert type(train) is type(test) is type(dev) is Subset
        assert len(dev) != 0 and len(train) != 0 and len(test) != 0
        assert len(dev) < len(train) > len(test)

if __name__ == '__main__':
    unittest.main()
