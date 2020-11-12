import unittest

from util import time_spent


class TestUtil(unittest.TestCase):
    def test_time_spent(self):
        self.assertEqual(time_spent(30), "30s")
        self.assertEqual(time_spent(100), " 1m 40s")
        self.assertEqual(time_spent(60*60+1), " 1h  1s")
        self.assertEqual(time_spent(4*60*60 + 61), " 4h  1m  1s")
        self.assertEqual(time_spent(24*60*60+ 4*60*60 + 60 + 1), " 1d  4h  1m  1s")

if __name__ == '__main__':
    unittest.main()