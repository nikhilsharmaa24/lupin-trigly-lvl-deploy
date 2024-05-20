import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

# Load the data
df_train = pd.read_csv('D:/Nikhil/hackathon/lupin/dataset/train.csv')
df_test = pd.read_csv('D:/Nikhil/hackathon/lupin/dataset/test.csv')

# Drop the candidate_id column
df_train = df_train.drop(columns=['candidate_id'])
df_test = df_test.drop(columns=['candidate_id'])

# Separate features and target variable from training data
X = df_train[['bad_cholestrol_lvl', 'total_cholestrol', 'good_cholestrol_lvl']]
y = df_train['triglyceride_lvl']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the SVR model
model_svr = SVR()

# Define hyperparameters grid for Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.1, 0.2, 0.5, 1.0]
}

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=model_svr, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best SVR model from Grid Search
best_model_svr = grid_search.best_estimator_

# Fit the best SVR model to the training data
best_model_svr.fit(X_train_scaled, y_train)

# Predict on the validation set using the best SVR model
y_val_pred_svr = best_model_svr.predict(X_val_scaled)

# Evaluate the best SVR model
mae_svr = mean_absolute_error(y_val, y_val_pred_svr)
print(f'Best Model - Mean Absolute Error (MAE) with Support Vector Regression: {mae_svr}')

# Save the model and scaler
#joblib.dump(best_model_svr, 'svr_model.pkl')
#joblib.dump(scaler, 'scaler.pkl')

pickle.dump(best_model_svr, open('svr_model.pkl', 'wb'))




