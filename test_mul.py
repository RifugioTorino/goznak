import unittest
from functools import reduce


def multiplicate(a):
    """
    :param a: Массив целых чисел ненулевой длины
    :return: Массив такой же длины, в котором на i-ом месте находится произведение всех чисел массива А,
             кроме числа, стоящего на i-ом месте.
    """
    assert len(a) > 0  # ensure non-emptiness
    assert all(map(lambda x: isinstance(x, int), a))  # ensure integer
    assert all(map(lambda x: x != 0, a))  # ensure non-zero values

    b = reduce(lambda x, y: x*y, a)
    return [b//x for x in a]


class TestM(unittest.TestCase):

    def test_empty(self):
        self.assertRaises(AssertionError, multiplicate, [])

    def test_int(self):
        inputs = [[1, 2, '3', 4], [0 + 1j], [5, 5.5, -5, 5]]
        for a in inputs:
            self.assertRaises(AssertionError, multiplicate, a)

    def test_value(self):
        inputs = [[1, 2, 3, 4], [5], [5, 5, -5, 5]]
        results = [[24, 12, 8, 6], [1], [-125, -125, 125, -125]]
        for a, b in zip(inputs, results):
            self.assertEqual(multiplicate(a), b)

    def test_zero(self):
        inputs = [[1, 2, 0, 4], [0], [5, 5, -0, 5]]
        for a in inputs:
            self.assertRaises(AssertionError, multiplicate, a)


if __name__ == '__main__':
    unittest.main()
