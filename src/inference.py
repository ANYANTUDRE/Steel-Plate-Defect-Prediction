import joblib
import os
import pandas as pd
import configs

model ='xgb'

def predict(model):
    test   = pd.read_csv(configs.TEST_SET).values
    sample = pd.read_csv(configs.SAMPLE_FILE)
    targets = ["Pastry", "Z_Scratch", "K_Scatch", 
               "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    
    predictions = None

    for fold in range(5):
        clf = joblib.load(os.path.join(configs.MODEL_OUTPUT, f"{model}_{fold}.pkl"))
        preds = clf.predict_proba(test)
        if fold == 0:
            predictions = preds
        else:
            predictions += preds
    predictions = predictions / 5

    sample[targets] = predictions
    print(sample.head())
    return sample


if __name__ == "__main__":
    #models = ["hist", "cat", "gbm", "lgbm", "xgb"]
    submission = predict(model)
    submission.to_csv(f"../output/xgb_submission.csv", index=False)