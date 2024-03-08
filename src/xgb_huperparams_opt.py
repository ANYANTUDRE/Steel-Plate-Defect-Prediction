import configs
import model_dispatcher
import pandas as pd 
import numpy as np
from sklearn import metrics
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb

cv_scores = []
 # read the training data with folds
train = pd.read_csv(configs.TRAIN_FOLDS)
targets = ["Pastry", "Z_Scratch", "K_Scatch", 
        "Stains", "Dirtiness", "Bumps", "Other_Faults"]
features = [c for c in train.columns.to_list() if c not in targets]

def objective(trial):
    params = {
        'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'gamma' : trial.suggest_float('gamma', 1e-9, 0.5),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'max_depth': trial.suggest_int('max_depth', 0, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),
    }
    
    params['booster'] = 'gbtree'
    params["verbosity"] = 0
    params['tree_method'] = "hist"

    for fold in range(5):
        df_train = train[train.kfold != fold].reset_index(drop=True)
        df_valid = train[train.kfold == fold].reset_index(drop=True)

        x_train = df_train.drop(targets+["kfold"], axis=1).values
        y_train = df_train[targets].values

        x_valid = df_valid.drop(targets+["kfold"], axis=1).values
        y_valid = df_valid[targets].values

        # initialize simple decision tree classifier from sklearn
        clf = xgb.XGBClassifier(**params)

        # fit the model on training data
        clf.fit(x_train, y_train)

        # create predictions for validation samples
        preds = clf.predict_proba(x_valid)

        # calculate roc auc
        auc = metrics.roc_auc_score(y_valid, preds, multi_class="ovr")
        cv_scores.append(auc)
    
    cv_evaluation = np.mean(cv_scores)
    return cv_evaluation

study_name = "steel_plate"
study = optuna.create_study(study_name=study_name, 
                            sampler=TPESampler(n_startup_trials=30, multivariate=True, seed=0),
                            direction="maximize")

study.optimize(objective, n_trials=100)
best_cls_params = study.best_params
best_value = study.best_value

print(f"best optmized accuracy: {best_value:0.5f}")
print(f"best hyperparameters: {best_cls_params}")