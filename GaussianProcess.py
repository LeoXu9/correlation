import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv('augmented_rgb_1.csv')

# Split input and output data
X = data.iloc[1:, [1,2,3,5]].values
y = data.iloc[1:, -2].values

# Standardize the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define kernel
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)

# Define K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameters for grid search
param_grid = {'kernel': [ConstantKernel(1.0, (1e-3, 1e4)) * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2)),
                         ConstantKernel(1.0, (1e-3, 1e4)) * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e2))],
              'alpha': [0.01, 0.1, 1, 10]}

# Perform grid search to optimize kernel and alpha
grid_search = GridSearchCV(gp, param_grid=param_grid, cv=kfold, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best R2 score
best_kernel = grid_search.best_estimator_.kernel
best_alpha = grid_search.best_estimator_.alpha
best_r2 = grid_search.best_score_

# Define Gaussian Process Regressor with best hyperparameters
gp = GaussianProcessRegressor(kernel=best_kernel, alpha=best_alpha)

# Fit the model on the training data
gp.fit(X_train, y_train)

# Predict the output for the testing data
y_pred = gp.predict(X_test)

# Evaluate the performance on the testing data
r2 = r2_score(y_test, y_pred)

# Print
print('Best kernel:', best_kernel)
print('Best alpha:', best_alpha)
print('Cross-validation R2 scores:', grid_search.cv_results_['mean_test_score'])
print('Mean cross-validation R2 score:', np.mean(grid_search.cv_results_['mean_test_score']))
print('Testing R2 score:', r2)
