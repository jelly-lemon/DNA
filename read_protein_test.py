import unittest
import numpy as np
from read_protein import to_vector, to_onehot_matrix, replace, convert_lines_to_onthot, merged_batch, to_number_array


class TestProtein(unittest.TestCase):
    def test_to_onehot_matrix(self):
        #
        # 合法输入
        #
        r1 = to_onehot_matrix('AC')
        r2 = np.zeros((1000, 20), dtype=np.uint8)
        r2[0][0] = 1
        r2[1][1] = 1
        self.assertTrue((r1 == r2).all(), msg="转换得到one-hot矩阵数据有误")
        self.assertEqual(r1.shape, r2.shape, msg="转换得到one-hot矩阵shape有误")

        r1 = to_onehot_matrix('AB') # 非法氨基酸字母
        r2 = np.zeros((1000, 20), dtype=np.uint8)
        r2[0][0] = 1
        self.assertTrue((r1 == r2).all(), msg="转换得到one-hot矩阵数据有误")
        self.assertEqual(r1.shape, r2.shape, msg="转换得到one-hot矩阵shape有误")

        #
        # 不合法输入
        #
        self.assertRaises(ValueError, to_onehot_matrix, "")
        self.assertRaises(ValueError, to_onehot_matrix, "\n")
        self.assertRaises(ValueError, to_onehot_matrix, "\r")

    def test_to_vector(self):
        #
        # 合法输入
        #
        r1 = to_vector('A') # 氨基酸字母
        r2 = np.zeros((1, 20), dtype=np.uint8)
        r2[0][0] = 1
        self.assertTrue((r1 == r2).all())

        r1 = to_vector('X') # 补全字母
        r2 = np.zeros((1, 20), dtype=np.uint8)
        self.assertTrue((r1 == r2).all())

        #
        # 不合法输入
        #
        self.assertRaises(ValueError, to_vector, 'B')   # 不存在的大写字母
        self.assertRaises(ValueError, to_vector, 'a')   # 小写字母

    def test_replace(self):
        # 合法输入
        str = "ABCDEF\n"
        str = replace(str)
        self.assertEqual(str, "AXCDEF\n")

    def test_convert_lines_to_onthot(self):
        lines = ["ABC\n", "DEF\n"]
        one_hot = convert_lines_to_onthot(lines)
        self.assertEqual(one_hot.shape, (2, 1000, 20))

    def test_merged_batch(self):
        batch_size = 32
        gen = merged_batch(batch_size)
        x, y = next(gen)
        self.assertEqual(x.shape, (batch_size, 1000, 20))
        self.assertEqual(y.shape, (batch_size, ))

    def test_to_number_array(self):
        r1 = to_number_array('ACDEF') # 长度为1000
        r2 = np.zeros((1000,))
        r2[0] = 1
        r2[1] = 2
        r2[2] = 3
        r2[3] = 4
        r2[4] = 5
        self.assertTrue((r1 == r2).all())

if __name__ == '__main__':
    unittest.main()