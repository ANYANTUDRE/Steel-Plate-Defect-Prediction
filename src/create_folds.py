import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import configs

if __name__ == "__main__":
    # import the dataset
    train = pd.read_csv(configs.TRAIN_SET)

    # create new column called kfold and fill it with -1
    train["kfold"] = -1

    # randomize the rows of the data
    train = train.sample(frac=1).reset_index(drop=True)

    targets = ["Pastry", "Z_Scratch", "K_Scatch", 
               "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    # fetch the 7 targets
    y = train[targets]

    

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=2024)

    for fold, (train_idx, val_idx) in enumerate(mskf.split(train, y)):
        train.loc[val_idx, "kfold"] = fold

    # save the new csv with kfold column
    train.to_csv("../input/train_folds.csv", index=False)
    print(train.head())
