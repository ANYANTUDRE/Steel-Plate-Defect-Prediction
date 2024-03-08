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
                             #n_estimators= 785,
                             tree_method='hist',
                             eta=0.06, 
                             #gamma=1, 
                             max_depth=5
                             ),    
    "xgb_tuned":    xgb.XGBClassifier(** {'grow_policy': 'lossguide', 'n_estimators': 343, 
                                          'learning_rate': 0.2953847324956156, 'gamma': 0.4810942725965306, 
                                          'subsample': 0.4741272004639706, 'colsample_bytree': 0.7033101340924858, 'max_depth': 7, 
                                          'min_child_weight': 5, 'reg_lambda': 2.8437928571268845e-07, 'reg_alpha': 30.216149925077563})    ,
    
    "lgbm": lgbm.LGBMClassifier(n_jobs=-1),
    'gbm': ensemble.GradientBoostingClassifier(),
    #'cat': catboost.CatBoostClassifier(verbose=False),
}