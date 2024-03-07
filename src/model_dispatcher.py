from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgbm
#import catboost

models = { 
    "hist": ensemble.HistGradientBoostingClassifier(learning_rate=0.16, l2_regularization=4,
                                                    max_iter=500, max_leaf_nodes=60, 
                                                    max_depth=13,
                                                    ),  # bon score, rivalise avec XGBoost

    #"extra": ensemble.ExtraTreesClassifier(), # nope, pas mieux que les autres gbdt

    "xgb": xgb.XGBClassifier(n_jobs=-1, 
                             #eta=0.01, 
                             #gamma=0.5, 
                             #max_depth=7
                             ),            # plutot pas mal
    "lgbm": lgbm.LGBMClassifier(n_jobs=-1),
    'gbm': ensemble.GradientBoostingClassifier(),
    #'cat': catboost.CatBoostClassifier(verbose=False),
}