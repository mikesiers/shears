"""shears : Decision tree pruning functions.

This module can be used to use various decision tree pruning functions to
determine whether or not a decision tree node should be pruned.

"""
import scipy.stats

def estimated_errors(num_errors, num_records, confidence=25):
  """Calculates estimated errors using the Clopper-Pearson method.

  This code has been appropriated from:
  https://gist.github.com/DavidWalz/8538435

  Args:
    num_errors (int): The number of errors.

    num_records (int): The number of records.

    confidence (int): The confidence level. For more information on what the
      confidence level is, see Chapter 4 of the book:
      C4.5 programs for machine learning.
      https://books.google.com.au/books?isbn=1558602380

  Returns:
    (float): The estimated number of errors.
  """

  # Raise exceptions if any of the inputs are nonsensical.
  if num_errors > num_records:
    raise ValueError('There cannot be more errors than records.')
  if num_errors < 0:
    raise ValueError('There cannot be a negative number of errors.')
  if num_records < 0:
    raise ValueError('There cannot be a negative number of records.')
  if confidence < 0 or confidence > 50:
    raise ValueError('Confidence must be between 0 and 50 (both inclusive).')

  num_correct = num_records - num_errors
  significance = 1 - confidence / 100
  upper_bound = scipy.stats.beta.ppf(significance, num_errors+1, num_correct)
  return upper_bound * num_records
