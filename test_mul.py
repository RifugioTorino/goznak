import unittest
from functools import reduce


def multiplicate(a):
    """
    :param a: Массив целых чисел ненулевой длины
    :return: Массив такой же длины, в котором на i-ом месте находится произведение всех чисел массива А,
             кроме числа, стоящего на i-ом месте.
    """
    assert len(a) > 0

    b = reduce(lambda x, y: x*y, a)
    return [b//x for x in a]


class TestM(unittest.TestCase):

    def test_empty(self):
        self.assertRaises(AssertionError, multiplicate, [])

    def test_value(self):
        self.assertEqual(multiplicate([1, 2, 3, 4]), [24, 12, 8, 6])


if __name__ == '__main__':
    unittest.main()
