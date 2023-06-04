"""
Module to oversampling imbalance dataset for classification

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""

import os 
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import (RandomOverSampler, 
                                    SMOTE,
                                    SMOTENC,
                                    BorderlineSMOTE,
                                    ADASYN)


class imbalanced_dataset():
    """
    Class for treating imbalanced datatset by oversampling the minority class or classes

    args: 
        (1) path_to_file (type:str) - location of data file; last column is taken as the target feature
        (2) path_to_save (type:str) - location to save new data file

    return: 
        (1) pandas.Dataframe of test set (pkl and/or csv)
        (2) pandas.Dataframe of oversampled training set (pkl and/or csv)
    """

    def __init__(self, path_to_file, path_to_save, target, features):
        self.path_to_save = path_to_save
        self.df_dataset = joblib.load(path_to_file) 

        # Define input and target variables
        if isinstance(features, list):
            self.features = features
        else:
            self.features = joblib.load(features) 

        self.target = target

        print('Name of target column: ', self.target)
        print('No. of exploratory features: ', len(self.features))


    def oversampling(self, stratify, technique, *args, **kwargs):
        """
        Oversample with one of the following techniques: 
            (a)'ros'
            (b)'smoothed_ros'
            (c)'smote'
            (d)'smote_nc'
            (e)'smote_borderline1'
            (f)'smote_borderline2' 
            (g)'adasyn'

        args:
            (1) stratify (type:bool) - whether to stratify the dataset while splitting
            (2) technique (type:str) - oversampling technique to use
            (3*) categorical_features (list) - list of indices to specify the position of categorical columns; this is only applicable when using 'smote_nc' method

        return: 
            (1) pandas.Dataframe of test set (pkl and/or csv)
            (2) pandas.Dataframe of oversampled training set (pkl and/or csv)
        """

        if stratify == True:
            # Stratified data splitting 
            self.sample_train, self.sample_test = train_test_split(
                                                                    self.df_dataset, 
                                                                    test_size = 0.2, 
                                                                    stratify = self.df_dataset[self.target].to_list()
                                                                    )


            print('No. of rows in training set before oversampling:', len(self.sample_train))


        elif stratify == False:
            # Unstratified data splitting 
            self.sample_train, self.sample_test = train_test_split(
                                                                    self.df_dataset, 
                                                                    test_size = 0.2
                                                                    )


            print('No. of rows in training set before oversampling:', len(self.sample_train))


        # Define x and y variables
        x = self.sample_train[self.features].values
        y = self.sample_train[self.target].values


        # Different oversampling techniques 
        if technique == 'ros':
            os = RandomOverSampler()


        elif technique == 'smoothed_ros':
            os = RandomOverSampler(shrinkage = 0.15)


        elif technique == 'smote':
            os = SMOTE()


        elif technique == 'smote_nc':
            self.categorical_features = kwargs.get('categorical_features')
            os = SMOTENC(categorical_features = categorical_features, k_neighbors = 5)


        elif technique == 'smote_borderline1':
            os = BorderlineSMOTE(k_neighbors = 3, m_neighbors = 15, kind = 'borderline-1')


        elif technique == 'smote_borderline2':
            os = BorderlineSMOTE(k_neighbors = 3, m_neighbors = 15, kind = 'borderline-2')


        elif technique == 'adasyn':
            os = ADASYN()


        # Fit oversampling technique to traininng set
        x_oversampled, y_oversampled = os.fit_resample(x, y)


        # Create pandas.Dataframe of the new training set
        self.oversampled_train = pd.concat([
                                            pd.DataFrame(data = x_oversampled), 
                                            pd.DataFrame(data = y_oversampled, columns = [self.target])], 
                                            axis = 1
                                            )


        # Ensure the column names are consistent with test data
        self.oversampled_train.columns = self.features + [self.target]
        self.sample_test = self.sample_test[self.features + [self.target]] 


        print('No. of rows in training set after oversampling:', len(self.oversampled_train))


        return self.oversampled_train, self.sample_test



    def scale_features(self):
        """
        Scale the features between 0 and 1
        """

        # Define scaling factor
        scaling = MinMaxScaler(feature_range = (0, 1)) #Range can be adjusted


        # Create pandas.Dataframe of scaled training set
        oversampled_train_features = pd.DataFrame(
                                            scaling.fit_transform(self.oversampled_train[self.features].values), 
                                            columns=self.oversampled_train[self.features].columns, 
                                            index=self.oversampled_train[self.features].index
                                            )


        # Create pandas.Dataframe of scaled test set using scaler as defined using trainig set
        sample_test_features = pd.DataFrame(
                                            scaling.fit(self.sample_test[self.features].values), 
                                            columns=self.sample_test[self.features].columns, 
                                            index=self.sample_test[self.features].index
                                            )


        # Concatenate exploratory features and target feature           
        self.oversampled_train = pd.concat([
                                            oversampled_train_features, 
                                            self.oversampled_train[self.target]], 
                                            axis = 1
                                            )

        self.sample_test = pd.concat([
                                    sample_test_features, 
                                    self.sample_test[self.target]],
                                    axis = 1
                                    )


        print('Features successfully scaled')


        return self.oversampled_train, self.sample_test



    def save(self, name, csv):
        """
        Save train and test sets

        args: 
            (1) name (type:str); a string to add to file name
            (2) csv (type:bool); whether to save as csv 

        return: 
            (1) pandas.Dataframe of test set (pkl and/or csv)
            (2) pandas.Dataframe of oversampled training set (pkl and/or csv)
        """

        # Whether to save as csv
        if csv == True:
            self.sample_train.to_csv(os.path.join(self.path_to_save, r'train_set_' +  str(name) + '.csv'))
            self.oversampled_train.to_csv(os.path.join(self.path_to_save, r'train_set_oversampled_' +  str(name) + '.csv'))
            self.sample_test.to_csv(os.path.join(self.path_to_save, r'test_set_' +  str(name) + '.csv'))

            print('Datasets successfully saved as: ', 'test_set_' +  str(name) + '.csv')


        joblib.dump(self.sample_train, os.path.join(self.path_to_save, r'train_set_' +  str(name) + '.pkl'))
        joblib.dump(self.oversampled_train, os.path.join(self.path_to_save, r'train_set_oversampled_' +  str(name) + '.pkl'))
        joblib.dump(self.sample_test, os.path.join(self.path_to_save, r'test_set_' +  str(name) + '.pkl'))


        print('Datasets successfully saved as: ', 'test_set_' +  str(name) + '.pkl')

