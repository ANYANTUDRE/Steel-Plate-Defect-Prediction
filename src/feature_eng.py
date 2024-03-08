import pandas as pd
import numpy as np
import configs
from sklearn.preprocessing import MinMaxScaler

def calculate_coordinate_range_features(data):
    data['X_Range'] = (data['X_Maximum'] - data['X_Minimum'])
    data['Y_Range'] =( data['Y_Maximum'] - data['Y_Minimum'])
    return data

def calculate_luminosity_range_feature(data):
    data['Luminosity_Range'] = (data['Maximum_of_Luminosity'] - data['Minimum_of_Luminosity'])
    return data

def calculate_size_ratio_features(data):
    data['Area_Perimeter_Ratio'] = data['Pixels_Areas'] / (data['X_Perimeter'] + data['Y_Perimeter'])
    return data


def save_train_test_eng(train, test):
    train.to_csv(f"../input/train_engineered.csv", index=False)
    test.to_csv(f"../input/test_engineered.csv", index=False)

if __name__ == "__main__":
    # import the dataset
    train = pd.read_csv(configs.TRAIN_FOLDS)
    test   = pd.read_csv(configs.TEST_SET)

    targets = ["Pastry", "Z_Scratch", "K_Scatch", 
               "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    features = [c for c in train.columns.to_list() if c not in targets]
    y_train = train[targets]

    """train = calculate_coordinate_range_features(train)
    test  = calculate_coordinate_range_features(test)

    train = calculate_luminosity_range_feature(train)
    test  = calculate_luminosity_range_feature(test)

    train = calculate_size_ratio_features(train)
    test  = calculate_size_ratio_features(test)"""

    train_sc = train.astype(np.float64)
    test_sc  = test.astype(np.float64)

    mmscaler = MinMaxScaler()
    mmscaler.fit(train_sc)

    train_sc[:] = mmscaler.transform(train_sc[features])
    test_sc[:] = mmscaler.transform(test_sc)

    train = pd.concat([train_sc, y_train], axis=0)

    save_train_test_eng(train, test)