"""
Module to manipulate data

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  

import os                                                                                                                                                                                  
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler


class data_manipulation():
    """
    Class created for data manipulation process

    args: 
        (1) path_to_file (type:str) - path to the file of interest

    return: 
        (1) pandas.Dataframe of manipulated data set
    """

    def __init__(self, path_to_file):
        self.df = joblib.load(path_to_file)



    def drop_col(self, cols=[]):
        """
        Drops unwanted columns

        args: 
            (1) columns (list) - list of columns to drop

        returns: 
            (1) pandas.Dataframe 
        """

        self.df = self.df.drop(cols_to_drop, axis = 1)


        return self.df



    def none2zero(self, cols=[]):
        """
        Convert None into zero

        args: 
            (1) columns (list) - list of columns 

        returns: 
            (1) pandas.Dataframe 
        """

        for c in cols:
            self.df[c] = self.df[c].map(lambda x: 0 if x == 'None' else x)


        return self.df



    def assign2integer(self, cols):
        """
        Assign type to integer

        args: 
            (1) columns (list) - list of columns 

        returns: 
            (1) pandas.Dataframe 
        """

        for c in cols:
            self.df[c] = self.df[c].astype(int)


        return self.df



    def single_entry_col(self):
        """
        Find single entry columns

        args: None
            
        returns: 
            (1) list of columns with single entry
        """

        singleentry_columns = list()

        for c in self.df.columns:
            self.df[c] = self.df[c].astype('float')

            if self.df[self.df[c].isnull()].shape[0] / float(len(self.df)) >= 1:
                empty_columns.append(c)

            elif self.df[c].nunique() <= 1:
                singleentry_columns.append(c)


        return singleentry_columns



    def empty_col(self):
        """
        Find empty columns

        args: None
            
        returns: 
            (1) list of empty columns
        """

        empty_columns = list()

        for c in self.df.columns:
            self.df[c] = self.df[c].astype('float')

            if self.df[self.df[c].isnull()].shape[0] / float(len(self.df)) >= 1:
                empty_columns.append(c)


        return empty_columns



    def OHE(self, categorical_cols):
        """
        One-hot-encoding of categorical columns

        args: 
            (1) categorical_cols (list) - list of categorical columns

        return: 
            (1) one-hot-encoded categorical features
        """

        df_ohe = pd.get_dummies(data = self.df, columns = categorical_cols, prefix_sep = '_ohe_', drop_first = False)

        ohe_cols = [i for i in df_ohe.columns if '_ohe_' in i]


        return ohe_cols
