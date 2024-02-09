import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression

# Descargo la data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

# Inputs
X_name = 'GDP per capita (USD)'
X = lifesat['GDP per capita (USD)'].values
# Labels
y_name = 'Life satisfaction'
y = lifesat['Life satisfaction'].values

# Visualization
lifesat.plot(
    kind='scatter',
    grid=True,
    x=X_name,
    y=y_name,
)
# plt.show()

# Linear model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Hacemos una predicción para Cyprus, con GDP 37655.2
