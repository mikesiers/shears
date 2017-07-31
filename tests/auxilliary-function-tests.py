import sys
sys.path.append('../')
from src.shears import estimated_errors
import unittest

class test_auxilliary_functions(unittest.TestCase):

  def test_estimated_errors(self):
    # Check that the correct output is produced for the three examples on page
    # 41 of C4.5 programs for machine learning:
    # (https://books.google.com.au/books?isbn=1558602380).
    self.assertEqual(round(estimated_errors(0, 6), 3), 1.238)
    self.assertEqual(round(estimated_errors(0, 9), 3), 1.285)
    self.assertEqual(round(estimated_errors(0, 1), 3), 0.750)

    # Check that an error is raised if num_errors > num_records.
    with self.assertRaises(ValueError):
      estimated_errors(5, 4)

    # Check that an error is raised if num_errors < 0.
    with self.assertRaises(ValueError):
      estimated_errors(-5, 4)

    # Check that an error is raised if num_records < 0.
    with self.assertRaises(ValueError):
      estimated_errors(5, -4)

    # Check that an error is raised if confidence is not in [0, 50]
    with self.assertRaises(ValueError):
      estimated_errors(5, -4, 60)

if __name__ == '__main__':
    unittest.main(exit=False)
