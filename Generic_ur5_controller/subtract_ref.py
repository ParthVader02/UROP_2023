import pandas as pd
import matplotlib.pyplot as plt

ref = pd.read_csv("row_data_ref.csv")
data = pd.read_csv("row_data.csv")

relative = data - ref
relative.plot()
plt.show()