"""
Module to calculate permutation importance

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  

import os                                                                                                                                                                                  
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn import metrics

from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


class permutation_importance_analysis():
    """
	Class to examine the permutation importance of exploratory features

    args: 
        (1) path_to_file (type:str) - location of the data file with features
        (2) path_to_save (type:str) - location to save new data files
        (3) path_to_features (type:str) - location of the features to use (e.g. those with multicollinearity reduced)
        (4) problem (type:str) - whether it is a 'classification' or 'regression' problem
    
    return: 
        (1) result of permutation analysis
    """

    def __init__(self, path_to_file, path_to_save, path_to_features, problem, *args, **kwargs):
        self.path_to_save = path_to_save
        self.sample_train = joblib.load(path_to_file) 
        self.RFE_features = joblib.load(path_to_features)

        # Last column taken as the target variable or classes
        self.features = self.sample_train.columns.values[:-1]
        self.target = self.sample_train.columns.values[-1]

        self.problem = problem


    def base_model(self, boosting_method):
        """
        Select the baseline model 

        Note: 
        For classification, multi-class models are defined as shown below
        This can be changed into a binary problem by changing the 'objective' to 'binary' for LGBMClassifier, or to 'binary:logistic' or 'binary:logitraw' for XGBClassifier (see description in links below)
        
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        https://xgboost.readthedocs.io/en/latest/parameter.html
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html

        args: 
            (1) boosting_method (type:str) - either 'lightGBM' or 'XGBoost'

        return: 
            (1) baseline model
        """

        if self.problem == 'classification':
            if boosting_method == 'lightGBM':
                self.estimator = LGBMClassifier(
                                                boosting_type = 'gbdt',
                                                objective = 'multiclass',
                                                importance_type = 'gain',
                                                max_depth = -1
                                                )


            elif boosting_method == 'XGBoost':
                    self.estimator = XGBClassifier(
                                                    objective = 'multi:softprob',
                                                    booster = 'gbtree',
                                                    importance_type = 'total_gain'
                                                    )


        elif self.problem == 'regression':
            if boosting_method == 'lightGBM':
                self.estimator = LGBMRegressor(
                                                boosting_type ='gbdt',
                                                importance_type = 'gain',
                                                max_depth = -1
                                                )



            elif boosting_method == 'XGBoost':
                self.estimator = XGBClassifier(
                                                objective = 'reg:squarederror',
                                                booster = 'gbtree',
                                                importance_type = 'total_gain'
                                                )


        return self.estimator



    def calculate(self):
        """
        Calculate the permutation importance 
        """

        # Train baseline model
        self.model = self.estimator.fit(
                                        self.sample_train[self.RFE_features], 
                                        self.sample_train[self.target].values.ravel()
                                        )

        # Define metric to use
        if self.problem == 'classification':
            self.scoring = 'f1_weighted'

        elif self.problem == 'regression':
            self.scoring = 'neg_root_mean_squared_error'


        # Number of times to permute a feature
        n_repeats = 10

        # Calculate permutation importance
        self.result = permutation_importance(
                                            self.model, 
                                            self.sample_train[self.RFE_features], 
                                            self.sample_train[self.target].ravel(), 
                                            scoring = self.scoring,
                                            n_repeats = n_repeats
                                            )

        joblib.dump(self.result, os.path.join(self.path_to_save, r'permutation_importance_result.pkl'))

        print('Permutation data saved as: "permutation_importance_result.pkl"')



    def permutation_plot(self, no_features = 1):
        """
        Generate permutation plot

        args: 
            (1) no_features (type:int) - number of top features to include in the plot

        return: 
            (1) figure of permutation importance plot
        """

        # List of feature names
        feature_list = list(self.model.feature_name_)


        # Indices of features basd on permutation importance 
        sorted_index = self.result.importances_mean.argsort()


        # Permutation importance plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 8))

        ax1.boxplot(
                    self.result.importances[sorted_index].T, 
                    vert = False,
                    labels = [feature_list[i] for i in sorted_index]
                    )

        ax2.boxplot(
                    self.result.importances[sorted_index][-no_features:].T, 
                    vert = False,
                    labels = [feature_list[i] for i in sorted_index][-no_features:]
                    )

        fontsize = 15

        ax1.set_xlabel('Reduction in performance metric', fontsize = fontsize)
        ax2.set_xlabel('Reduction in performance metric', fontsize = fontsize)
        ax1.set_ylabel('Feature number in order of importance', fontsize = fontsize)

        fig.tight_layout()
        plt.show()

        fig.savefig(os.path.join(self.path_to_save, r'permutation_plot.png'), dpi = 300, bbox_inches="tight")
        
        print('Permutation plot saved as: "permutation_plot.png"')


