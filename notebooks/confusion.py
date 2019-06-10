import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import colors
from sklearn.metrics import confusion_matrix

def confusion_matrix_dataframe(Y_true, Y_pred):
  cm = confusion_matrix(np.argmax(Y_true.values, axis=1), np.argmax(Y_pred, axis=1))
  return pd.DataFrame(cm, columns=Y_true.columns, index=Y_true.columns)

def plot_confusion_matrix(cmdf, cmap='YlGnBu'):
  # Uses PowerNorm in order to get a more even swath of heatmap color spectrum
  sns.set(rc={'figure.figsize': (10, 8)})
  ax = sns.heatmap(data=cmdf, annot=True, fmt='d', square=True, norm=colors.PowerNorm(gamma=2/5), cmap=cmap)
  ax.set_xlabel('Predicted label')
  ax.set_ylabel('True label')
  return ax