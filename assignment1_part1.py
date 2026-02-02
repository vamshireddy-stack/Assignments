import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np.random.seed(12)
n_samples = 200

X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
print(X.shape)
# Cubic function y=x3−0.5x2+2x+noise
y = 0.12 * X[:,0]**3 + 0.5 * X[:, 0]**2 + 2 * X[:, 0] + np.random.normal(0, 2, n_samples)

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Simple Linear Regression Model, train the model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Let's test the model
y_pred_linear = linear_model.predict(X_test)

# Error calculation
linear_mse = mean_squared_error(y_test, y_pred_linear)
#Lets create a plot for Linear model

plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred_linear, color='red', label='Linear Prediction')
plt.title(f'Linear Regression\nMSE = {linear_mse:.2f}')
plt.legend()
plt.savefig("D:\AI_ML_OPS\Projects\linear_model.png", dpi=300, bbox_inches='tight')

# Polynomial Regression Model, transform the features first
degree_values = [2, 3, 5, 50]

plt.style.use("bmh")
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.patch.set_facecolor("#7E89EF")

# Sorting the test data for better visualization, without it train_test_split shuffles the data and the plot takes points from the Shuffle.
sort_index = np.argsort(X_test[:,0])
X_test_sort = X_test[sort_index]
y_test_sort = y_test[sort_index]

for ax, degree in zip(axs, degree_values):

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    # Let's transform the training data
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test_sort)
    

    # Build our model & train the model
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # Test the model
    y_pred_poly = poly_model.predict(X_test_poly)

    # MSE error
    poly_mse = mean_squared_error(y_test_sort, y_pred_poly)

    # plot all the results into a single fig
    ax.scatter(X_test, y_test, color='blue', label='Actual Data')
    ax.plot(X_test_sort, y_pred_poly, color="red", label=f'Polynomial Degree {degree} prediction')
    ax.set_title(f'Poly plot for degree {degree}\nMSE = {poly_mse:.2f} ')
    ax.legend()
    plt.tight_layout()

plt.savefig("D:\AI_ML_OPS\Projects\poly_model_over_underfitting.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

plt.show()