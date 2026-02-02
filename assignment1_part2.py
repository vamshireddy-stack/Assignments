import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("D:/AI_ML_OPS/Projects/auto-mpg.csv")
x = df.drop(["mpg", "car name", "origin"], axis = 1)
y = df["mpg"]

# print(x.info(), y.info())
# Converting non-numeric columns to numeric, and handling '?' other characters by <error='coerce'> to NaN
x["horsepower"] = pd.to_numeric(x["horsepower"], errors='coerce')
x["cylinders"] = pd.to_numeric(x["cylinders"], errors='coerce')
x["displacement"] = pd.to_numeric(x["displacement"], errors='coerce')
x["weight"] = pd.to_numeric(x["weight"], errors='coerce')
x["acceleration"] = pd.to_numeric(x["acceleration"], errors='coerce')
x["model year"] = pd.to_numeric(x["model year"], errors='coerce')
# Our linear models don't handle NaN values, so filling NaN with mean of the column
x = x.fillna(x.mean())
# print(x.info())

#Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


degree = 15
# model multiple linear regression
linear_model = Pipeline([
    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
    ("linear", LinearRegression())
])
linear_model.fit(x_train, y_train)

y_pred_linear = linear_model.predict(x_test)

# Calculate MSE
mse_linear = mean_squared_error(y_test, y_pred_linear)
print("MSE for multi rg model:", mse_linear)


# ridge, linear, elasticnet regularization test with 4 values
alphas = [0.1, 1, 5]

# Empty lists for plot1
mse_train_ridge = []
mse_test_ridge = []
mse_train_lasso = []
mse_test_lasso = []
mse_train_elastic = []
mse_test_elastic = []

# empty list for plot2
ridge_coefs =[]
lasso_coefs =[]
elastic_coefs =[]


for alpha in alphas:
    # model ridge regression
    ridge_model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])
    ridge_model.fit(x_train, y_train)

    # Find y value for train and test values for ridge model & append to empty MSE list for ridge
    y_train_ridge = ridge_model.predict(x_train)
    y_test_ridge = ridge_model.predict(x_test)
    mse_train_ridge.append(mean_squared_error(y_train, y_train_ridge))
    mse_test_ridge.append(mean_squared_error(y_test, y_test_ridge))
    ridge_coefs.append(ridge_model.named_steps["ridge"].coef_)

    # lasso model
    lasso_model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha))
    ])
    lasso_model.fit(x_train, y_train)

    # Find y value for train and test values for lasso model & append to MSE list for lasso
    y_train_lasso = lasso_model.predict(x_train)
    y_test_lasso = lasso_model.predict(x_test)
    mse_train_lasso.append(mean_squared_error(y_train, y_train_lasso))
    mse_test_lasso.append(mean_squared_error(y_test, y_test_lasso))
    lasso_coefs.append(lasso_model.named_steps["lasso"].coef_)

    # elasticnet model
    elasticnet_model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("elasticnet", ElasticNet(alpha=alpha, l1_ratio=0.5))
    ])
    elasticnet_model.fit(x_train, y_train)

    # Find y value for train and test values for elastic model & append to MSE list for elasticnet
    y_train_elastic = elasticnet_model.predict(x_train)
    y_test_elastic = elasticnet_model.predict(x_test)
    mse_train_elastic.append(mean_squared_error(y_train, y_train_elastic))
    mse_test_elastic.append(mean_squared_error(y_test, y_test_elastic))
    elastic_coefs.append(elasticnet_model.named_steps["elasticnet"].coef_)



# Plot 1 "Training vs testing error plot for different regularizationstrength" using the above lists
plt.style.use("bmh")
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
fig.patch.set_facecolor("#7E89EF")

models = ["Ridge", "Lasso", "ElasticNet"]
mse_train = [mse_train_ridge, mse_train_lasso, mse_train_elastic]
mse_test = [mse_test_ridge, mse_test_lasso, mse_test_elastic]

for i, name in enumerate(models):
    ax = axs[i]
    ax.plot(alphas, mse_train[i], marker='o', label='Train MSE')
    ax.plot(alphas, mse_test[i], marker='o', label='Test MSE')
    ax.set_title(f"{name} - Train vs Test mse")
    ax.set_xlabel("alpha [0.1, 1, 5]")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

axs[0].set_ylabel("MSE")


plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


# Plot 2 "coefficient shrinkage path plot showing how feature weights change as regularization increases "
ridge_alpha_coef = np.array(ridge_coefs)
lasso_alpha_coef = np.array(lasso_coefs)
elastic_alpha_coef = np.array(elastic_coefs)

features = np.arange(ridge_alpha_coef.shape[1]) #acts as address for polynomial features
colors = ['#1f77b4', "#1dcb40", "#f06868"]
markers = ['o','s','^']

fig, axes = plt.subplots(3, 1, figsize=(10,5), sharex=True)
models_data = [ridge_alpha_coef, lasso_alpha_coef, elastic_alpha_coef]
model_names = ["Ridge", "Lasso", "ElasticNet"]

for i, ax in enumerate(axes):
    data = models_data[i]

    # Now plot 3 lines in each sub-plot
    # x value, y value from features(index of ridge_alpha_coef), models_data
    for alpha_idx, alpha_val in enumerate(alphas):
        ax.plot(features, data[alpha_idx], 
                label=f'Alpha={alpha_val}',
                marker=markers[alpha_idx],color=colors[alpha_idx],
                linewidth=1.2)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(model_names[i], fontsize=10, fontweight='bold')
    ax.set_ylabel('Coefficient Weight', fontsize=8)
    ax.legend(title="Regualarization alpha", loc='upper right')
    ax.grid(True, linestyle=":", alpha=0.5)


plt.xlabel('Feature Index', fontsize=10)
plt.tight_layout()
plt.savefig("D:\AI_ML_OPS\Projects\poly_model_over_underfitting.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

plt.show()
# print(mse_test)