"""shears : Decision tree pruning functions.

This module can be used to use various decision tree pruning functions to
determine whether or not a decision tree node should be pruned.

"""
import scipy.stats

def estimated_errors(num_errors, num_records, confidence=25):
  """Calculates estimated errors using the Clopper-Pearson method.

  This code has been appropriated from:
  https://gist.github.com/DavidWalz/8538435
  """
  num_correct = num_records - num_errors
  significance = 1 - confidence / 100
  upper_bound = scipy.stats.beta.ppf(significance, num_errors+1, num_correct)
  return upper_bound * num_records
