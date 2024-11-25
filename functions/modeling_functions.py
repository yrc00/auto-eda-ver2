"""
This is the Overview page

"""

###################################### import ######################################

# library
import streamlit as st
import pandas as pd
import numpy as np
import gettext
import os

import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

###################################### set  ######################################

# language setting
locale_path = os.path.join(os.path.dirname(__file__), 'locales')
translator = gettext.translation('base', localedir=locale_path, languages=[st.session_state.language], fallback=True)
translator.install()
_ = translator.gettext

##################################### Model ######################################

# randomforest classifier
def rfc(df, target, test_size,criterion, max_depth, min_samples_split, min_samples_leaf, max_features, n_estimator, random_state):
    df_encoded = pd.get_dummies(df, drop_first=True)

    # split data
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    # train test split and fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    rfc = RandomForestClassifier(criterion=criterion, 
                                 max_depth = max_depth,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf, 
                                 max_features=max_features,
                                 n_estimators=n_estimator, 
                                 random_state=random_state)
    
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    top15 = pd.Series(rfc.feature_importances_, index=X.columns).nlargest(15)
    
    return accuracy, precision, recall, f1, top15
    
# randomforest regressor
def rfr(df, target, test_size,criterion, max_depth, min_samples_split, min_samples_leaf, max_features, n_estimator, random_state):
    df_encoded = pd.get_dummies(df, drop_first=True)

    # split data
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    # train test split and fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    rfr = RandomForestRegressor(criterion=criterion, 
                                 max_depth = max_depth,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf, 
                                 max_features=max_features,
                                 n_estimators=n_estimator, 
                                 random_state=random_state)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)

    # evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    top15 = pd.Series(rfr.feature_importances_, index=X.columns).nlargest(15)
    return mse, rmse, mae, r2, top15

# decision tree classifier
def dtc(df, target, test_size, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, random_state):
    df_encoded = pd.get_dummies(df, drop_first=True)
    if max_features == "None":
        max_features = None

    # split data
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    # train test split and fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    dtc = DecisionTreeClassifier(criterion=criterion, 
                                 max_depth=max_depth, 
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf, 
                                 max_features=max_features, 
                                 random_state=random_state)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    top15 = pd.Series(dtc.feature_importances_, index=X.columns).nlargest(15)
    return accuracy, precision, recall, f1, top15

# decision tree regressor
def dtr(df, target, test_size, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, random_state):
    df_encoded = pd.get_dummies(df, drop_first=True)

    # split data
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    # train test split and fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    dtr = DecisionTreeRegressor(criterion=criterion, 
                                max_depth=max_depth, 
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, 
                                max_features=max_features, 
                                random_state=random_state)
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_test)

    # evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    top15 = pd.Series(dtr.feature_importances_, index=X.columns).nlargest(15)
    return mse, rmse, mae, r2, top15

# xgboost classifier
def xgc(df, target, test_size, eta, max_depth, objective, eval_metric, random_state):
    df_encoded = pd.get_dummies(df, drop_first=True)

    # split data
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    # train test split and fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if (eta is None) and (max_depth is None) and (objective is None) and (eval_metric is None):
        xgc = XGBClassifier(random_state=random_state)
    else:
        xgc = XGBClassifier(eta=eta, 
                            max_depth=max_depth, 
                            objective=objective, 
                            eval_metric=eval_metric, 
                            random_state=random_state)
    xgc.fit(X_train, y_train)
    y_pred = xgc.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    top15 = pd.Series(xgc.feature_importances_, index=X.columns).nlargest(15)
    return accuracy, precision, recall, f1, top15

# xgboost regressor
def xgr(df, target, test_size, eta, max_depth, objective, eval_metric, random_state):
    df_encoded = pd.get_dummies(df, drop_first=True)
    if eval_metric == "None":
        eval_metric = None

    # split data
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    # train test split and fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if (eta is None) and (max_depth is None) and (objective is None) and (eval_metric is None):
        xgr = XGBRegressor(random_state=random_state)
    else:
        xgr = XGBRegressor(eta=eta, 
                        max_depth=max_depth, 
                        objective=objective, 
                        eval_metric=eval_metric,  
                        random_state=random_state)
    xgr.fit(X_train, y_train)
    y_pred = xgr.predict(X_test)

    # evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    top15 = pd.Series(xgr.feature_importances_, index=X.columns).nlargest(15)
    return mse, rmse, mae, r2, top15

##################################### Optuna ######################################

# plot feature importance
def plot_feature_importances(top15, model_name):
    if not top15.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top15.values, y=top15.index)
        plt.title(f"Top 15 Feature Importance for {model_name}")
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        st.pyplot(plt.gcf())
    else:
        st.write(f"No feature importance to display for {model_name}")

# Random Forest Regressor
def rfr_objective(trial, X_train, y_train, X_test, y_test, metric):
    # Suggest hyperparameters for the Random Forest Regressor
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse'])  
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    random_state = trial.suggest_int('random_state', 0, 100)

    # Create a Random Forest model with the sampled hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        max_features=max_features,
        random_state=random_state  # Use the suggested random state
    )

    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate the chosen metric
    if metric == 'MSE':
        score = mean_squared_error(y_test, y_pred)
    elif metric == 'RMSE':
        score = np.sqrt(mean_squared_error(y_test, y_pred))
    elif metric == 'MAE':
        score = mean_absolute_error(y_test, y_pred)
    elif metric == 'R2':
        score = r2_score(y_test, y_pred)

    return score, random_state

def randomforest_regression_optuna(df, target, n_trials=50, metric='MSE'):
    # Encode categorical features and drop missing values
    df_encoded = pd.get_dummies(df, drop_first=True).dropna()

    # Split features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set direction based on the chosen metric
    direction = 'minimize' if metric in ['MSE', 'RMSE', 'MAE'] else 'maximize'

    # Optuna study for hyperparameter optimization
    study = optuna.create_study(direction=direction)
    study.optimize(lambda trial: rfr_objective(trial, X_train, y_train, X_test, y_test, metric)[0], n_trials=n_trials)

    # Get the best parameters
    best_params = study.best_params

    # Extract the best random state from the best trial
    best_random_state = study.trials[study.best_trial.number].params['random_state']

    # Remove 'random_state' from best_params
    best_params.pop('random_state', None)

    best_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        criterion=best_params['criterion'],
        max_features=best_params['max_features'],
        random_state=best_random_state  # Use the best random state
    )

    # Train the best model
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    # Evaluate the best model
    mse_best = mean_squared_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    mae_best = mean_absolute_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)

    # Feature importance
    feature_importances = best_model.feature_importances_
    feature_names = X.columns

    # Top 15 feature importance
    top15 = pd.Series(feature_importances, index=feature_names).nlargest(15)

    return best_random_state, best_params, mse_best, rmse_best, mae_best, r2_best, top15

# Random Forest Classifier
def rfc_objective(trial, X_train, y_train, X_test, y_test, metric):
    # Set hyperparameters
    criterion = trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"])
    max_features = trial.suggest_categorical('max_features', ["sqrt", "log2"])
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    random_state = trial.suggest_int('random_state', 0, 100)

    # Random Forest Classifier model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        criterion=criterion,
        max_features=max_features
    )

    # Model fit
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate based on metric
    if metric == "accuracy":
        score = accuracy_score(y_test, y_pred)
    elif metric == "precision":
        score = precision_score(y_test, y_pred)
    elif metric == "recall":
        score = recall_score(y_test, y_pred)
    elif metric == "f1":
        score = f1_score(y_test, y_pred)

    return score, random_state  # Return the score and random state

def randomforest_classification_optuna(df, target, n_trials=50, metric='accuracy'):
    # Dataset encoding and cleaning
    df_encoded = pd.get_dummies(df, drop_first=True).dropna()

    # Split features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: rfc_objective(trial, X_train, y_train, X_test, y_test, metric)[0], n_trials=n_trials)

    # Retrieve the best hyperparameters
    best_params = study.best_params
    
    # Extract the best random state from the best trial
    best_random_state = study.trials[study.best_trial.number].params['random_state']

    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=best_random_state,
        criterion=best_params['criterion'],
        max_features=best_params['max_features']
    )

    # Train the best model
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    # Evaluate the best model
    accuracy_best = accuracy_score(y_test, y_pred_best)
    precision_best = precision_score(y_test, y_pred_best)
    recall_best = recall_score(y_test, y_pred_best)
    f1_best = f1_score(y_test, y_pred_best)

    # Feature importance
    feature_importances = best_model.feature_importances_
    feature_names = X.columns

    # Top 15 feature importance
    top15 = pd.Series(feature_importances, index=feature_names).nlargest(15)

    return best_random_state, best_params, accuracy_best, precision_best, recall_best, f1_best, top15

# decision tree regressor
def dtr_objective(trial, X_train, y_train, X_test, y_test, metric):
    # Set hyperparameters
    max_features = trial.suggest_categorical('max_features', ["sqrt", "log2", None]) 
    criterion = trial.suggest_categorical('criterion', ["squared_error", "friedman_mse", "absolute_error", "poisson"]) 
    max_depth = trial.suggest_int('max_depth', 5, 50) 
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10) 
    random_state = trial.suggest_int('random_state', 0, 100) 
     
    # Decision tree regressor model
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        criterion=criterion,
        max_features=max_features
    )

    # Model fit
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate based on metric selected
    if metric == "MSE":
        score = mean_squared_error(y_test, y_pred)
    elif metric == "RMSE":
        score = np.sqrt(mean_squared_error(y_test, y_pred))
    elif metric == "MAE":
        score = mean_absolute_error(y_test, y_pred)
    elif metric == "R2":
        score = r2_score(y_test, y_pred)

    return score, random_state  # Return the score and random state


def decisiontree_regression_optuna(df, target, n_trials=50, metric='MSE'):
    # Dataset encoding and cleaning
    df_encoded = pd.get_dummies(df, drop_first=True).dropna()

    # Split features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set optimization direction based on metric
    direction = 'minimize' if metric in ['MSE', 'RMSE', 'MAE'] else 'maximize'

    # Optuna study for hyperparameter optimization
    study = optuna.create_study(direction=direction)
    study.optimize(lambda trial: dtr_objective(trial, X_train, y_train, X_test, y_test, metric)[0], n_trials=n_trials)

    # Retrieve the best hyperparameters
    best_params = study.best_params
    
    # Extract the best random state from the best trial
    best_random_state = study.trials[study.best_trial.number].params['random_state']

    best_model = DecisionTreeRegressor(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=best_random_state,  # Use the best random state
        criterion=best_params['criterion'],
        max_features=best_params['max_features']
    )

    # Train the best model
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    # Evaluate the model using the specified metrics
    mse_best = mean_squared_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    mae_best = mean_absolute_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)

    # Feature importance
    feature_importances = best_model.feature_importances_
    feature_names = X.columns

    # Top 15 feature importance
    top15 = pd.Series(feature_importances, index=feature_names).nlargest(15)

    return best_random_state, best_params, mse_best, rmse_best, mae_best, r2_best, top15

# Decision Tree Classifier
def dtc_objective(trial, X_train, y_train, X_test, y_test, metric):
    # Set hyperparameters
    criterion = trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"]) 
    max_features = trial.suggest_categorical('max_features', ["sqrt", "log2", None]) 
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20) 
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    random_state = trial.suggest_int('random_state', 0, 100)

    # Decision tree classifier model
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        criterion=criterion,
        max_features=max_features
    )

    # Model fit
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate based on metric selected
    if metric == "accuracy":
        score = accuracy_score(y_test, y_pred)
    elif metric == "precision":
        score = precision_score(y_test, y_pred, average='weighted')
    elif metric == "recall":
        score = recall_score(y_test, y_pred, average='weighted')
    elif metric == "f1":
        score = f1_score(y_test, y_pred, average='weighted')

    return score, random_state  # Return the score and random state


def decisiontree_classifier_optuna(df, target, n_trials=50, metric='accuracy'):
    # Dataset encoding and cleaning
    df_encoded = pd.get_dummies(df, drop_first=True).dropna()

    # Split features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: dtc_objective(trial, X_train, y_train, X_test, y_test, metric)[0], n_trials=n_trials)

    # Retrieve the best hyperparameters
    best_params = study.best_params
    
    # Extract the best random state from the best trial
    best_random_state = study.trials[study.best_trial.number].params['random_state']

    best_model = DecisionTreeClassifier(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=best_random_state,
        criterion=best_params['criterion'],
        max_features=best_params['max_features']
    )

    # Train the best model
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    # Evaluate the model using the specified metric
    accuracy_best = accuracy_score(y_test, y_pred_best)
    precision_best = precision_score(y_test, y_pred_best, average='weighted')
    recall_best = recall_score(y_test, y_pred_best, average='weighted')
    f1_best = f1_score(y_test, y_pred_best, average='weighted')

    # Feature importance
    feature_importances = best_model.feature_importances_
    feature_names = X.columns

    # Top 15 feature importance
    top15 = pd.Series(feature_importances, index=feature_names).nlargest(15)

    # Return the best random state, parameters, and evaluation metrics
    return best_random_state, best_params, accuracy_best, precision_best, recall_best, f1_best, top15

# XGBoost regressor
def xgr_objective(trial, X_train, y_train, X_test, y_test, metric):
    # Set hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    objective = trial.suggest_categorical('objective', ['reg:squarederror', 'reg:squaredlogerror'])
    eval_metric = trial.suggest_categorical('eval_metric', ['rmse', 'mae'])
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
    gamma = trial.suggest_float('gamma', 0.0, 1.0)
    random_state = trial.suggest_int('random_state', 0, 100)

    # XGBoost Regressor model
    model = XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective=objective,
        eval_metric=eval_metric,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state  # Use random_state
    )

    # Model fit
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate based on metric selected
    if metric == "MSE":
        score = mean_squared_error(y_test, y_pred)
    elif metric == "RMSE":
        score = np.sqrt(mean_squared_error(y_test, y_pred))
    elif metric == "MAE":
        score = mean_absolute_error(y_test, y_pred)
    elif metric == "R2":
        score = r2_score(y_test, y_pred)

    return score, random_state  # Return score and random_state

def xgboost_regression_optuna(df, target, n_trials=50, metric='MSE'):
    # Dataset encoding and cleaning
    df_encoded = pd.get_dummies(df, drop_first=True).dropna()

    # Split features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set optimization direction based on metric
    direction = 'minimize' if metric in ['MSE', 'RMSE', 'MAE'] else 'maximize'

    # Optuna study for hyperparameter optimization
    study = optuna.create_study(direction=direction)
    study.optimize(lambda trial: xgr_objective(trial, X_train, y_train, X_test, y_test, metric)[0], n_trials=n_trials)

    # Retrieve the best hyperparameters
    best_params = study.best_params
    
    # Extract the best random state from the best trial
    best_random_state = study.trials[study.best_trial.number].params['random_state']

    best_model = XGBRegressor(
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        objective=best_params['objective'],
        eval_metric=best_params['eval_metric'],
        n_estimators=best_params['n_estimators'],
        min_child_weight=best_params['min_child_weight'],
        gamma=best_params['gamma'],
        random_state=best_random_state  # Use the best random state
    )

    # Train the best model
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    # Evaluate the model using the specified metrics
    mse_best = mean_squared_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    mae_best = mean_absolute_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)

    # Feature importance
    feature_importances = best_model.feature_importances_
    feature_names = X.columns

    # Top 15 feature importance
    top15 = pd.Series(feature_importances, index=feature_names).nlargest(15)

    return best_random_state, best_params, mse_best, rmse_best, mae_best, r2_best, top15

# XGBoost classifier
def xgbc_objective(trial, X_train, y_train, X_test, y_test, metric):
    # Set hyperparameters
    eta = trial.suggest_float('eta', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_float('gamma', 0.0, 1.0)
    random_state = trial.suggest_int('random_state', 0, 100)

    # XGBoost Classifier model
    model = XGBClassifier(
        eta=eta,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        eval_metric='logloss',  
        random_state=random_state
    )

    # Model fit
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate based on metric selected
    if metric == "accuracy":
        score = accuracy_score(y_test, y_pred)
    elif metric == "precision":
        score = precision_score(y_test, y_pred, average='weighted')
    elif metric == "recall":
        score = recall_score(y_test, y_pred, average='weighted')
    elif metric == "f1":
        score = f1_score(y_test, y_pred, average='weighted')

    return score, random_state

def xgboost_classifier_optuna(df, target, n_trials=50, metric='accuracy'):
    # Dataset encoding and cleaning
    df_encoded = pd.get_dummies(df, drop_first=True).dropna()

    # Split features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: xgbc_objective(trial, X_train, y_train, X_test, y_test, metric)[0], n_trials=n_trials)

    # Retrieve the best hyperparameters
    best_params = study.best_params
    
    # Extract the best random state from the best trial
    best_random_state = study.trials[study.best_trial.number].params['random_state']

    best_model = XGBClassifier(
        eta=best_params['eta'],
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        eval_metric='logloss',
        random_state=best_random_state
    )

    # Train the best model
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    # Evaluate the model using the specified metric
    accuracy_best = accuracy_score(y_test, y_pred_best)
    precision_best = precision_score(y_test, y_pred_best, average='weighted')
    recall_best = recall_score(y_test, y_pred_best, average='weighted')
    f1_best = f1_score(y_test, y_pred_best, average='weighted')

    # Feature importance
    feature_importances = best_model.feature_importances_
    feature_names = X.columns

    # Top 15 feature importance
    top15 = pd.Series(feature_importances, index=feature_names).nlargest(15)

    # Return the best parameters, evaluation metrics, and random_state
    return best_random_state, best_params, accuracy_best, precision_best, recall_best, f1_best, top15

def optuna_supervised(df, target):
    with st.container(border=True):
        st.markdown("**Select Model**")

        # model selection
        col1, col2 = st.columns([1, 2])
        with col1:
            model_type = st.selectbox("Model Type", ["Classification", "Regression"])
            st.session_state.model_type = model_type

        with col2:
            models = st.multiselect(
                "Models",
                ["Random Forest", "Decision Tree", "XGBoost"],
                ["Random Forest", "Decision Tree", "XGBoost"]
            )
        
        # criterion selection

        col3, col4 = st.columns([1, 1])
        with col3:
            if model_type == "Classification":
                metric = st.selectbox("Metric", ["accuracy", "precision", "recall", "f1"])
            else:
                metric = st.selectbox("Metric", ["MSE", "RMSE", "MAE", "R2"])
        with col4:
            trials = st.number_input("Number of Trials", value=50, min_value=1, max_value=100, step=1)

        save_data = st.checkbox("Save the result of learning", value=True)

    start = st.button("Start Learn")

    if start:
        st.divider()
        st.markdown("**Results**")
        results = []

        # classification case
        if model_type == "Classification":
            for model_name in models:
                if model_name == "Random Forest":
                    best_random_state, best_params, accuracy_best, precision_best, recall_best, f1_best, top15 = randomforest_classification_optuna(df, target, trials, metric=metric)
                elif model_name == "Decision Tree":
                    best_random_state, best_params, accuracy_best, precision_best, recall_best, f1_best, top15 = decisiontree_classifier_optuna(df, target, trials, metric=metric)
                elif model_name == "XGBoost":
                    best_random_state, best_params, accuracy_best, precision_best, recall_best, f1_best, top15 = xgboost_classifier_optuna(df, target, trials, metric=metric)

                results.append([model_name, accuracy_best, precision_best, recall_best, f1_best, best_random_state, best_params])

                if save_data:
                    st.session_state.supervised_results.append(
                        {
                            'metrics': {"Model": model_name, "Test Size": 0.2, "Random State": best_random_state, "Type": "Classification", "Accuracy": accuracy_best, "Precision": precision_best, "Recall": recall_best, "F1": f1_best},
                            'parameters': best_params,
                            'top15': top15
                        }
                    )
                with st.expander(f"**Top 15 Feature Importance for {model_name}**"):
                    plot_feature_importances(top15, model_name)

            # convert results to dataframe and display
            results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Random State", "Best Params"])
            st.dataframe(results_df.style.highlight_max(["Accuracy", "Precision", "Recall", "F1"], axis=0))
    
        else:
            for model_name in models:
                if model_name == "Random Forest":
                    best_random_state, best_params, mse_best, rmse_best, mae_best, r2_best, top15 = randomforest_regression_optuna(df, target, trials, metric=metric)
                elif model_name == "Decision Tree":
                    best_random_state, best_params, mse_best, rmse_best, mae_best, r2_best, top15 = decisiontree_regression_optuna(df, target, trials, metric=metric)
                elif model_name == "XGBoost":
                    best_random_state, best_params, mse_best, rmse_best, mae_best, r2_best, top15 = xgboost_regression_optuna(df, target, trials, metric=metric)

                results.append([model_name, mse_best, rmse_best, mae_best, r2_best, best_random_state, best_params])

                if save_data:
                    st.session_state.supervised_results.append(
                        {
                            'metrics': {"Model": model_name, "Test Size": 0.2, "Random State": best_random_state, "Type": "Regression", "MSE": mse_best, "RMSE": rmse_best, "MAE": mae_best, "R2": r2_best},
                            'parameters': best_params,
                            'top15': top15
                        }
                    )

                with st.expander(f"**Top 15 Feature Importance for {model_name}**"):
                    plot_feature_importances(top15, model_name)

            # convert results to dataframe and display
            results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "MAE", "R2", "Random State", "Best Params"])
            st.dataframe(results_df.style.highlight_min(["MSE", "RMSE", "MAE", "R2"], axis=0))

    # return model_type
    return model_type

##################################### simple learning ######################################

# randomforest classifier
def randomforest_classifier_options():
    max_features = st.selectbox("max_features", ["option", "int"],
                                placeholder="default is option, select max features from option or int", key="rf_max_features")
    col5, col6, col7 = st.columns([1, 1, 1])

    criterion = col5.selectbox("criterion", ["gini", "entropy", "log_loss"],
                               placeholder="default is gini, select criterion from gini or entropy", key="rf_criterion")
    max_depth = col6.number_input("max_depth", value=None, min_value=1, max_value=100, step=1,
                                  placeholder="default is None, select max depth from 1 to 100", key="rf_max_depth")
    min_samples_split = col7.number_input("min_samples_split", value=2, min_value=2, max_value=100, step=1,
                                          placeholder="default is 2, select min samples split from 2 to 100", key="rf_min_samples_split")
    min_samples_leaf = col5.number_input("min_samples_leaf", value=1, min_value=1, max_value=100, step=1,
                                         placeholder="default is 1, select min samples leaf from 1 to 100", key="rf_min_samples_leaf")
    
    if max_features == "option":
        max_features = col6.selectbox("max_features", ["sqrt", "log2"],
                                      placeholder="default is sqrt, select max features from sqrt or log2", key="rf_max_features_option")
    else:
        max_features = col6.number_input("max_features", value=1.0, min_value=0.1, max_value=1.0, step=0.1,
                                         placeholder="default is 1.0, select max features from 0.1 to 1.0", key="rf_max_features_int")
    
    n_estimator = col7.number_input("n_estimator", value=100, min_value=1, max_value=1000, step=1,
                                    placeholder="default is 100, select n estimator from 1 to 1000", key="rf_n_estimator")
    
    return criterion, max_depth, min_samples_split, min_samples_leaf, max_features, n_estimator

# randomforest regressor
def randomforest_regressor_options():
    max_features = st.selectbox("max_features", ["option", "int"],
                                placeholder="default is option, select max features from option or int", key="rfr_max_features")
    col5, col6, col7 = st.columns([1, 1, 1])

    criterion = col5.selectbox("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                               placeholder="default is squared_error, select criterion", key="rfr_criterion")
    max_depth = col6.number_input("max_depth", value=None, min_value=1, max_value=100, step=1,
                                  placeholder="default is None, select max depth", key="rfr_max_depth")
    min_samples_split = col7.number_input("min_samples_split", value=2, min_value=2, max_value=100, step=1,
                                          placeholder="default is 2", key="rfr_min_samples_split")
    min_samples_leaf = col5.number_input("min_samples_leaf", value=1, min_value=1, max_value=100, step=1,
                                         placeholder="default is 1", key="rfr_min_samples_leaf")
    
    if max_features == "option":
        max_features = col6.selectbox("max_features", ["sqrt", "log2"],
                                      placeholder="default is sqrt", key="rfr_max_features_option")
    else:
        max_features = col6.number_input("max_features", value=1.0, min_value=0.1, max_value=1.0, step=0.1,
                                         placeholder="default is 1.0", key="rfr_max_features_int")
    
    n_estimator = col7.number_input("n_estimator", value=100, min_value=1, max_value=1000, step=1,
                                    placeholder="default is 100", key="rfr_n_estimator")
    
    return criterion, max_depth, min_samples_split, min_samples_leaf, max_features, n_estimator

# decisiontree classifier
def decisiontree_classifier_options():
    max_features = st.selectbox("max_features", ["option", "int"],
                                placeholder="default is option", key="dtc_max_features")
    col5, col6, col7 = st.columns([1, 1, 1])

    criterion = col5.selectbox("criterion", ["gini", "entropy", "log_loss"],
                               placeholder="default is gini", key="dtc_criterion")
    max_depth = col6.number_input("max_depth", value=None, min_value=1, max_value=100, step=1,
                                  placeholder="default is None", key="dtc_max_depth")
    min_samples_split = col7.number_input("min_samples_split", value=2, min_value=2, max_value=100, step=1,
                                          placeholder="default is 2", key="dtc_min_samples_split")
    min_samples_leaf = col5.number_input("min_samples_leaf", value=1, min_value=1, max_value=100, step=1,
                                         placeholder="default is 1", key="dtc_min_samples_leaf")
    
    if max_features == "option":
        max_features = col6.selectbox("max_features", ["None", "sqrt", "log2"],
                                      placeholder="default is None", key="dtc_max_features_option")
    else:
        max_features = col6.number_input("max_features", value=1.0, min_value=0.1, max_value=1.0, step=0.1,
                                         placeholder="default is 1.0", key="dtc_max_features_int")
    
    return criterion, max_depth, min_samples_split, min_samples_leaf, max_features

# decisiontree regressor
def decisiontree_regressor_options():
    max_features = st.selectbox("max_features", ["option", "int"],
                                placeholder="default is option", key="dtr_max_features")
    col5, col6, col7 = st.columns([1, 1, 1])

    criterion = col5.selectbox("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                               placeholder="default is squared_error", key="dtr_criterion")
    max_depth = col6.number_input("max_depth", value=None, min_value=1, max_value=100, step=1,
                                  placeholder="default is None", key="dtr_max_depth")
    min_samples_split = col7.number_input("min_samples_split", value=2, min_value=2, max_value=100, step=1,
                                          placeholder="default is 2", key="dtr_min_samples_split")
    min_samples_leaf = col5.number_input("min_samples_leaf", value=1, min_value=1, max_value=100, step=1,
                                         placeholder="default is 1", key="dtr_min_samples_leaf")
    
    if max_features == "option":
        max_features = col6.selectbox("max_features", ["None", "sqrt", "log2"],
                                      placeholder="default is None", key="dtr_max_features_option")
    else:
        max_features = col6.number_input("max_features", value=1.0, min_value=0.1, max_value=1.0, step=0.1,
                                         placeholder="default is 1.0", key="dtr_max_features_int")
    
    return criterion, max_depth, min_samples_split, min_samples_leaf, max_features

# xgboost options
def xgboost_options():
    col5, col6 = st.columns([1, 1])

    eta = col5.number_input("eta", value=0.3, min_value=0.1, max_value=1.0, step=0.1,
                            placeholder="default is 0.3", key="xgb_eta")
    max_depth = col6.number_input("max_depth", value=6, min_value=1, max_value=100, step=1,
                                  placeholder="default is 6", key="xgb_max_depth")
    objective = st.selectbox("objective", ["reg:squarederror", "reg:squaredlogerror", "reg:logistic", "reg:pseudohubererror", "reg:absoluteerror", "reg:quantileerror",
                                           "binary:logistic", "binary:logitraw", "binary:hinge", "multi:softmax", "multi:softprob", "rank:pairwise", "rank:ndcg", "rank:map"],
                            placeholder="default is reg:squarederror", key="xgb_objective")
    eval_metric = st.selectbox("eval_metric", ["rmse", "mae", "logloss", "error", "merror", "mlogloss", "auc", "None"],
                               placeholder="default is rmse", key="xgb_eval_metric")
    
    return eta, max_depth, objective, eval_metric

# simple supervised learning
def simple_supervised(df, target):
    with st.container(border=True):
        st.markdown("**Select Model**")

        # model selection
        col1, col2 = st.columns([1, 2])
        with col1:
            model_type = st.selectbox("Model Type", ["Classification", "Regression"])
            st.session_state.model_type = model_type

        with col2:
            models = st.multiselect(
                "Models",
                ["Random Forest", "Decision Tree", "XGBoost"],
                ["Random Forest", "Decision Tree", "XGBoost"]
            )
        
        col3, col4 = st.columns([1, 1])
        # test size
        with col3: 
            test_size = st.number_input("Test Size", value=0.3, min_value=0.1, max_value=0.5, step=0.1, 
                                        placeholder="default is 0.3, select test size from 0.1 to 0.5")
        # random state
        with col4: 
            random_state = st.number_input("Random State", value=42, min_value=0, max_value=100, step=1,
                                            placeholder="default is 42, select random state from 0 to 100")
        
        more_options = st.checkbox("View more options")

    # Initialize default values
    d_criterion, d_max_depth, d_min_samples_split, d_min_samples_leaf, d_max_features = None, None, 2, 1, None

    # Initialize default values
    d_criterion, d_max_depth, d_min_samples_split, d_min_samples_leaf, d_max_features = None, None, 2, 1, None

    if more_options:
        if "Random Forest" in models:
            with st.container(border=True):
                st.markdown("**RandomForest**")
                if model_type == "Classification":
                    r_criterion, r_max_depth, r_min_samples_split, r_min_samples_leaf, r_max_features, r_n_estimator = randomforest_classifier_options()
                else: 
                    r_criterion, r_max_depth, r_min_samples_split, r_min_samples_leaf, r_max_features, r_n_estimator = randomforest_regressor_options()
        if "Decision Tree" in models:
            with st.container(border=True):
                st.markdown("**Decision Tree**")
                if model_type == "Classification":
                    d_criterion, d_max_depth, d_min_samples_split, d_min_samples_leaf, d_max_features = decisiontree_classifier_options()
                else: 
                    d_criterion, d_max_depth, d_min_samples_split, d_min_samples_leaf, d_max_features = decisiontree_regressor_options()
        if "XGBoost" in models:
            with st.container(border=True):
                st.markdown("**XGBoost**")
                x_eta, x_max_depth, x_objective, x_eval_metric = xgboost_options()
    else:
        if "Random Forest" in models:
            if model_type == "Classification":
                r_criterion, r_max_depth, r_min_samples_split, r_min_samples_leaf, r_max_features, r_n_estimator = "gini", None, 2, 1, "sqrt", 100
            else:
                r_criterion, r_max_depth, r_min_samples_split, r_min_samples_leaf, r_max_features, r_n_estimator = "squared_error", None, 2, 1, 1.0, 100
        if "Decision Tree" in models:
            if model_type == "Classification":
                d_criterion, d_max_depth, d_min_samples_split, d_min_samples_leaf, d_max_features = "gini", None, 2, 1, None
            else:
                d_criterion, d_max_depth, d_min_samples_split, d_min_samples_leaf, d_max_features = "squared_error", None, 2, 1, None
        if "XGBoost" in models:
            x_eta, x_max_depth, x_objective, x_eval_metric = 0.3, 6, "reg:squarederror", "logloss"

    start = st.button("Start Learn")

    if start:
        st.divider()
        st.markdown("**Results**")
        results = []

        # classification case
        if model_type == "Classification":
            for model_name in models:
                if model_name == "Random Forest":
                    accuracy, precision, recall, f1, top15 = \
                        rfc(df, target, test_size, r_criterion, r_max_depth, r_min_samples_split, r_min_samples_leaf, r_max_features, r_n_estimator, random_state)
                    parameters = {"criterion": r_criterion, "max_depth": r_max_depth, "min_samples_split": r_min_samples_split, "min_samples_leaf": r_min_samples_leaf, "max_features": r_max_features, "n_estimators": r_n_estimator}
                elif model_name == "Decision Tree":
                    accuracy, precision, recall, f1, top15 = \
                        dtc(df, target, test_size, d_criterion, d_max_depth, d_min_samples_split, d_min_samples_leaf, d_max_features, random_state)
                    parameters = {"criterion": d_criterion, "max_depth": d_max_depth, "min_samples_split": d_min_samples_split, "min_samples_leaf": d_min_samples_leaf, "max_features": d_max_features}
                elif model_name == "XGBoost":
                    accuracy, precision, recall, f1, top15 = \
                        xgc(df, target, test_size, x_eta, x_max_depth, x_objective, x_eval_metric, random_state)
                    parameters = {"eta": x_eta, "max_depth": x_max_depth, "objective": x_objective, "eval_metric": x_eval_metric}                        
                results.append([model_name, accuracy, precision, recall, f1])

                # store the result with metadata
                st.session_state.supervised_results.append(
                    {
                        "metrics": {"Model": model_name, "Test Size": test_size, "Random State": random_state, "Type": "Classification", "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1},
                        "parameters": parameters,
                        "top15": top15
                    }
                )

                with st.expander(f"**Top 15 Feature Importance for {model_name}**"):
                    plot_feature_importances(top15, model_name)

            # convert results to dataframe and display
            results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
            st.dataframe(results_df.style.highlight_max(["Accuracy", "Precision", "Recall", "F1"], axis=0))
            
        else:
            for model_name in models:
                if model_name == "Random Forest":
                    mse, rmse, mae, r2, top15 = \
                        rfr(df, target, test_size, r_criterion, r_max_depth, r_min_samples_split, r_min_samples_leaf, r_max_features, r_n_estimator, random_state)
                    parameters = {"criterion": r_criterion, "max_depth": r_max_depth, "min_samples_split": r_min_samples_split, "min_samples_leaf": r_min_samples_leaf, "max_features": r_max_features, "n_estimators": r_n_estimator}
                elif model_name == "Decision Tree":
                    mse, rmse, mae, r2, top15 = \
                        dtr(df, target, test_size, d_criterion, d_max_depth, d_min_samples_split, d_min_samples_leaf, d_max_features, random_state)
                    parameters = {"criterion": d_criterion, "max_depth": d_max_depth, "min_samples_split": d_min_samples_split, "min_samples_leaf": d_min_samples_leaf, "max_features": d_max_features}
                elif model_name == "XGBoost":
                    mse, rmse, mae, r2, top15 = \
                        xgr(df, target, test_size, x_eta, x_max_depth, x_objective, x_eval_metric, random_state)
                    parameters = {"eta": x_eta, "max_depth": x_max_depth, "objective": x_objective, "eval_metric": x_eval_metric}
                results.append([model_name, mse, rmse, mae, r2])

                # store the result with metadata
                st.session_state.supervised_results.append(
                    {
                        "metrics": {"Model": model_name, "Test Size": test_size, "Random State": random_state, "Type": "Regression", "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2},
                        "parameters": parameters,
                        "top15": top15
                    }
                )

                with st.expander(f"**Top 15 Feature Importance for {model_name}**"):
                    plot_feature_importances(top15, model_name)

            # convert results to dataframe and display
            results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "MAE", "R2"])
            st.dataframe(results_df.style.highlight_min(["MSE", "RMSE", "MAE"], axis=0))
    return model_type
        
##################################### supervised learning page ######################################

def show_learning(df, target):
    optuna = st.toggle("Use Optuna", False)

    if optuna:
        model_type = optuna_supervised(df, target)

    else:
        model_type = simple_supervised(df, target)

    return model_type

def show_all_results(model_type):
    if st.session_state.supervised_results:
        st.markdown("**All Results**")

        # Add a selectbox to choose the sorting metric
        metrics_options = []
        if model_type == "Classification":
            metrics_options = ["Accuracy", "Precision", "Recall", "F1"]
        else:  # Regression
            metrics_options = ["MSE", "RMSE", "MAE", "R2"]
        
        sort_by = st.selectbox("Sort by", metrics_options, index=0)

        # Convert stored results to a DataFrame for sorting
        metrics_df_list = []
        for result in st.session_state.supervised_results:
            metrics = result["metrics"]
            metrics_df_list.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_df_list)

        # Handle sorting based on model type
        if model_type == "Classification":
            if sort_by in metrics_df.columns:
                metrics_df.sort_values(by=sort_by, ascending=False, inplace=True)
            else:
                st.error(f"Metric '{sort_by}' is not available in the results.")
        else:
            if sort_by in metrics_df.columns:
                metrics_df.sort_values(by=sort_by, ascending=True, inplace=True)
            else:
                st.error(f"Metric '{sort_by}' is not available in the results.")
            
        st.dataframe(metrics_df)

        # Display metrics and parameters for each result
        for idx, result in enumerate(st.session_state.supervised_results):
            with st.expander(f"**Result {idx+1} - {result['metrics']['Model']} ({result['metrics']['Type']})**"):
                # Display metrics
                metrics_df = pd.DataFrame(result["metrics"], index=[0])
                st.markdown("**Metrics**")
                st.dataframe(metrics_df)

                # Display parameters
                parameters_df = pd.DataFrame(result["parameters"], index=[0])
                st.markdown("**Parameters**")
                st.dataframe(parameters_df)

                # Plot top 15 feature importance (if available)
                if not result["top15"].empty:
                    st.markdown("**Top 15 Feature Importance**")
                    plot_feature_importances(result["top15"], result["metrics"]["Model"])
        if st.button("Reset Results"):
            st.session_state.supervised_results = []
    else:
        st.warning(_("No results to display"))

# supervised learning
def supervised(df, target):
    # title
    st.markdown("### Supervised Learning")

    if 'supervised_results' not in st.session_state:
        st.session_state.supervised_results = []

    # Learning tab
    with st.container(border=True):
        tab1, tab2 = st.tabs(["Learning", "Results"])
        with tab1:
            model_type = show_learning(df, target)

        with tab2:
            show_all_results(model_type)

##################################### Clustering ######################################

# clustering pairplot
@st.cache_data
def clustering_pairplot(df):
    return sns.pairplot(df, hue="Cluster", palette="viridis", markers='o')

# Clustinerg - DBSCAN
def DBSCAN_model(df, eps, min_samples):
    # standardize the data
    df_scaled = StandardScaler().fit_transform(df)
    
    # clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(df_scaled)
    fig = clustering_pairplot(df)
    st.pyplot(fig)    

# Clustering - KMeans
def KMeans_model(df, target, n_clusters, random_state):
    # split data
    X = df.drop(target, axis=1)
    y = df[target]

    # standardize the data
    X_scaled = StandardScaler().fit_transform(X)

    # clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    fig = clustering_pairplot(df)
    st.pyplot(fig)

# clustering
def clustering(df, target):
    # clustering
    st.markdown("### Clustering")
    
    col1, col2 = st.columns([3, 1])
        
    # select model
    with col1:
        model = st.multiselect("Model", ["DBSCAN", "KMeans"], ["DBSCAN", "KMeans"])
        
    # start button
    with col2:
        start = st.button("Start Clustering")

    # DBSCAN
    if "DBSCA in model":
        with st.container(border=True):
            st.markdown("**DBSCAN**")

            col1, col2= st.columns(2)

            # epsilon
            with col1:
                eps = st.number_input(
                    "Epsilon", 
                    value=0.5, 
                    step=0.1, 
                    format="%.1f", 
                    placeholder="default is 0.5"
                )

            # min samples            
            with col2:
                min_samples = st.number_input(
                    "Min Samples", 
                    value=df.shape[1]+1, 
                    step=1, 
                    placeholder="default is number of columns + 1"
                )

            # start dbscan
            if start:
                with st.expander("**DBSCAN Pairplot**"):
                    DBSCAN_model(df.dropna(), eps, min_samples)            
    
    # KMeans
    if "KMeans" in model:
        with st.container(border=True):
            st.markdown("**KMeans**")

            col1, col2 = st.columns(2)

            # number of clusters
            with col1:
                n_clusters = st.number_input(
                    "Number of Clusters", 
                    value=3, 
                    step=1, 
                    placeholder="default is 3"
                )
            
            # random state
            with col2:
                random_state = st.number_input(
                    "Random State", 
                    value=42, 
                    step=1, 
                    placeholder="default is 42"
                )

            # start kmeans
            if start:
                with st.expander("**KMeans Pairplot**"):
                    KMeans_model(df.dropna(), target, n_clusters, random_state)

##################################### PCA ######################################

# PCA plot
@st.cache_data
def pca_plot(df, target, n_components):
    # split data
    X = df.drop(target, axis=1)
    y = df[target]

    # standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    # explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # pca result
    pca_df = pd.DataFrame(data=pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df[target] = y.values

    # plot
    unique_target = pca_df[target].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_target)))

    fig = plt.figure()
    for color, target_value in zip(colors, unique_target):
        idx = pca_df[target] == target_value
        plt.scatter(pca_df.loc[idx, 'PC1'], pca_df.loc[idx, 'PC2'], c=[color], label=target_value)
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    
    return fig, explained_variance_ratio


# PCA
def pca(df, target):
    # PCA
    st.markdown("### PCA")
    
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        if 'n_components' in st.session_state:
            n_components = st.session_state.n_components
        else:
            n_components = 2
        
        # set number of components
        with col1:
            n_components = st.number_input("Number of Components", value=n_components, step=1, placeholder="default is 2")
        
        # start button
        with col2:
            start = st.button("Start PCA")
        
        # start PCA
        if start:
            fig, explained_variance_ratio = pca_plot(df.dropna(), target, n_components)
            st.pyplot(fig)
            
            st.write("Explained Variance Ratio")
            st.write(explained_variance_ratio)

            st.session_state.n_components = n_components

