# %%

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

from skutils import get_coef_t_values_linear_model

# %%
X, y = load_iris(return_X_y=True)
y = X[:, 0]
X = X[:, 1:4]

# %%
model = LinearRegression()
model.fit(X, y)

# %%
t_values, p_values = get_coef_t_values_linear_model(X, y, model)

# %%
model.coef_

# %%
t_values

# %%
p_values

# %%
