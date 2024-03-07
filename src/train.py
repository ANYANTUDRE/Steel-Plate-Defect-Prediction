import os
import argparse
import configs
import model_dispatcher
import joblib
import pandas as pd 
from sklearn import metrics

def run(fold, model):
    # read the training data with folds
    train = pd.read_csv(configs.TRAIN_FOLDS)

    targets = ["Pastry", "Z_Scratch", "K_Scatch", 
               "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    features = [c for c in train.columns.to_list() if c not in targets]

    df_train = train[train.kfold != fold].reset_index(drop=True)
    df_valid = train[train.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(targets+["kfold"], axis=1).values
    y_train = df_train[targets].values

    x_valid = df_valid.drop(targets+["kfold"], axis=1).values
    y_valid = df_valid[targets].values

    # initialize simple decision tree classifier from sklearn
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict_proba(x_valid)

    # calculate roc auc
    auc = metrics.roc_auc_score(y_valid, preds, multi_class="ovr")
    print(f"Fold--->{fold}, AUC score={auc}")
    #print(f"Confusion Matrix:\n {metrics.confusion_matrix(y_valid, preds)}")

    # save the model
    joblib.dump(clf, os.path.join(configs.MODEL_OUTPUT, f"{model}_{fold}.pkl"))

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their types
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(fold=args.fold, model=args.model)