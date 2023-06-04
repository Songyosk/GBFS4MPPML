"""
Module to featurize CIFs using JarvisCFID

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""

import os
import pandas as pd
import numpy as np
import pymatgen as pmg
import timeout_decorator
import pathlib
import joblib

from matminer.featurizers.structure import JarvisCFID 


class use_cfid():
    """
    Class to generate JarvisCFID features using CIFs

    args: 
        (1) name_of_parent_folder (type:str) - must match name of directory containing the 'cif' folder
        (2) csv (type:bool) - whether to save data as csv
        
    return: 
        (1) pandas.Dataframe of CFID features (pkl and/or csv)
    """

    def __init__(self, name_of_parent_folder, csv):
        self.name_of_parent_folder = name_of_parent_folder
        self.csv = csv

        self.cur_dir = pathlib.Path().resolve()
        self.directory = os.path.join(self.cur_dir, 'retrieved_data', self.name_of_parent_folder, 'cifs')
        self.directory_2 = os.path.join('retrieved_data', self.name_of_parent_folder)
        
        self.jarvis = JarvisCFID()

        print(self.jarvis)



    #@timeout_decorator.timeout(100, timeout_exception = TimeoutError) #100 seconds timer
    def descriptor(self, cif):
        """
        Apply CFID descriptor

        args: 
            (1) CIFs 

        return: 
            (2) CFID features
        """

        struc = pmg.Structure.from_file(os.path.join(self.directory, cif))  

        output = self.jarvis.featurize(struc)

        return output



    def featurize(self):
        """
        Create features using the 'descriptor()' function with a time limit of 100 seconds
        """

        # Create a list of cifs
        files = [f for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f))]

        cif_files = [f for f in files if os.path.splitext(f)[1] == '.cif']

        print('No. of CIFs: ', len(cif_files))


        # Featurise
        jarvis_features, cif_success, cif_timedout, cif_error = list(), list(), list(), list()

        for cif in cif_files:
            try:
                cif_features = self.descriptor(cif)
                jarvis_features.append(cif_features)
                cif_success.append(os.path.splitext(cif)[0])

                print('Success with ', cif) 


            except TimeoutError:
                cif_timedout.append(os.path.splitext(cif)[0])

                print('Timeout with ', cif) 
                pass

            except: 
                cif_error.append(os.path.splitext(cif)[0])

                print('Error with ', cif) 
                pass


        print('no. of data saved:', len(cif_success))
        print('no. of errors:', len(cif_error))
        print('no. of time-outs:', len(cif_timedout))


        # Create pandas.Dataframe of the complied data
        if len(cif_success) != 0:
            #Features
            df_cif = pd.DataFrame(jarvis_features) 

            #CIF IDs
            df_index = pd.DataFrame(cif_success) 
            df_index = df_index.rename(columns={0: 'task_id'}) 

            #Concat two dataframes
            df = pd.concat([df_index, df_cif], axis=1)
            df['task_id'] = df['task_id'].map(lambda x: x.rstrip('.cif'))

            df = df.set_index('task_id')


        # CIF with erros
        df_error = pd.DataFrame(cif_error) 
        df_error = df_error.rename(columns={0: 'task_id'}) 


        # CIF that timed out
        df_timedout = pd.DataFrame(cif_timedout) 
        df_timedout = df_timedout.rename(columns={0: 'task_id'}) 


        # Save data
        if len(cif_success) != 0:
            #print('List of cif_success')
            #print(cif_success)

            df_cif_success = pd.DataFrame({'cif_success':cif_success})
            joblib.dump(df_cif_success, os.path.join(self.directory_2,r'cif_success' +  str(self.name_of_parent_folder) + '.pkl'))

            joblib.dump(df, os.path.join(self.directory_2,r'CFID_features_' +  str(self.name_of_parent_folder) + '.pkl'))

            print('Successfully saved data as: ', 'CFID_features_' +  str(self.name_of_parent_folder) + '.pkl')

            if self.csv == True:
                df_cif_success.to_csv(os.path.join(self.directory_2,r'cif_success' +  str(self.name_of_parent_folder) + '.csv'), index = False)
                df.to_csv(os.path.join(self.directory_2,r'CFID_features_' +  str(self.name_of_parent_folder) + '.csv'), index = False)

                print('Successfully saved data as: ', 'CFID_features_' +  str(self.name_of_parent_folder) + '.csv')


        joblib.dump(df_error, os.path.join(self.directory_2,r'CFID_error_' +  str(self.name_of_parent_folder) + '.pkl'))
        joblib.dump(df_timedout, os.path.join(self.directory_2,r'CFID_timedout_' +  str(self.name_of_parent_folder) + '.pkl'))

        if self.csv == True:
            df_error.to_csv(os.path.join(self.directory_2,r'CFID_error_' +  str(self.name_of_parent_folder) + '.csv'), index = False)
            df_timedout.to_csv(os.path.join(self.directory_2,r'CFID_timedout_' +  str(self.name_of_parent_folder) + '.csv'), index = False)



