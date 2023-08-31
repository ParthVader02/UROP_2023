import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

# Read in the confusion matrix
confusion_matrix = pd.read_csv("confusion_matrix.csv", header=0, index_col=0)

sns.heatmap(confusion_matrix, fmt='g', cmap='Greens', xticklabels=True, yticklabels=True)
plt.show()