import pandas as pd
import xgboost as xgb
import optuna
import os
from kaggle import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error

os.chdir('C://Users//j_chr//OneDrive//Desktop//Projects//essay grader//')

# Downloading data from kaggle

api = KaggleApi()
api.authenticate()

api.dataset_download_files('mazlumi/ielts-writing-scored-essays-dataset', path = '.', unzip=True)

df = pd.read_csv("ielts_writing_dataset.csv")


# Prepare the data
essays = df['Essay']
questions = df['Question']
y = df['Overall']

X = [q + " " + e for q, e in zip(questions, essays)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Convert data to DMatrix (XGBoost format)
dtrain = xgb.DMatrix(X_train_vectorized, label=y_train)
dtest = xgb.DMatrix(X_test_vectorized, label=y_test)

# Set parameters for the XGBoost model
params = {
    'objective': 'reg:squarederror',  # For regression tasks
    'max_depth': 8,
    'learning_rate': 0.1291,
    'subsample': 0.6886,  # default bagging in XGBoost
    'colsample_bytree': 0.6075,  # feature sampling in XGBoost
    'alpha': 0,  # L2 regularization
    'lambda': 7,
    'eval_metric': 'rmse',
    'seed': 42
}

# Train the model
model = xgb.train(params, dtrain, num_boost_round=435, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)

# Predict
y_pred = model.predict(dtest)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

model.save_model('xgb_essay_grader.json')

# Hyper parameter tuning

# def objective(trial):
#     param = {
#         'objective': 'reg:squarederror',  # Regression with squared error loss
#         'max_depth': trial.suggest_int('max_depth', 4, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Sample ratio of training data
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Feature sampling
#         'alpha': trial.suggest_int('alpha', 0, 10),  # L1 regularization
#         'lambda': trial.suggest_int('lambda', 1, 10),  # L2 regularization
#         'eval_metric': 'rmse',  # Evaluation metric
#         'seed': 42
#     }
    
#     # Train the model using XGBoost
#     model = xgb.train(param, dtrain, num_boost_round=trial.suggest_int('iterations', 100, 500), 
#                       evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)
    
#     # Predict and calculate RMSE
#     preds = model.predict(dtest)
#     rmse = mean_squared_error(y_test, preds, squared=False)  # RMSE calculation
#     return rmse

# # Create and optimize the study
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# # Print the best hyperparameters and RMSE
# print('Best hyperparameters: ', study.best_params)
# print('Best RMSE: ', study.best_value)

# Best hyperparameters:  {'max_depth': 8, 'learning_rate': 0.12909475451443067, 'subsample': 0.6886491811844366, 'colsample_bytree': 0.6074946529907628, 'alpha': 0, 'lambda': 7, 'iterations': 290}
# Best RMSE:  0.819096037785157