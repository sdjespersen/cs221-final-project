import collections
import numpy as np
import pandas as pd

def accuracy_breakdown(Ytrue, Ypred):
  """
  Given the true labels and the predictions, returns
  the category-by-category accuracy scores.
  """
  # Ytrue should be a pandas df
  totals = Ytrue.sum()
  guessed, correct = collections.Counter(), collections.Counter()
  for target, pred in zip(np.argmax(Ytrue.values, axis=1), np.argmax(Ypred, axis=1)):
    guessed[pred] += 1
    if target == pred:
      correct[target] += 1
  correct = pd.Series({Ytrue.columns[k]: correct[k] for k in correct})
  guessed = pd.Series({Ytrue.columns[k]: guessed[k] for k in guessed})
  stats = pd.DataFrame({
    'totals': totals,
    'correct': correct,
    'guessed': guessed,
  })
  stats.correct = stats.correct.fillna(0).astype(int)
  stats.guessed = stats.guessed.fillna(0).astype(int)
  stats['recall'] = (correct / totals).fillna(0)
  stats['precision'] = (correct / guessed).fillna(0)
  return stats
