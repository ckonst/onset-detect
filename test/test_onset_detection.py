# -*- coding: utf-8 -*-

import librosa as lb
import unittest

from model.hyperparameters import DSP
import onset_detect

class OnsetDetectionTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
         cls.input_sig, cls.fs = lb.load('./audio/pop_shuffle.wav')
         cls.dsp = DSP()
         print(cls.input_sig, cls.fs)

    def test_get_lmfs(self):
        lmfs = onset_detect.get_lmfs(self.fs, self.input_sig, self.dsp.W, self.dsp.stride)
        assert lmfs.size > 0

    def test_superflux(self):
        pass

    def test_get_onset_frames(self):
        pass

    def test_get_onset_times(self):
        pass

    def test_neural_onsets(self):
        pass

    def test_create_click_track(self):
        pass

if __name__ == '__main__':
    unittest.main()
