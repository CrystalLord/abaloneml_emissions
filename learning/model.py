import os

from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

import learning

class Model:

    def __init__(self, cleaner):
        """Create a Model Object.

        Args:
            df: Data frame with all of the data
            cleaner: DataCleaner object used
        """
        self.frames = cleaner.frames
        self.data_cleaner = cleaner
        self.k_folds()

    def k_folds(self):
        # Removing old version of the file training data will be read from
        dataFile = 'training.csv'
        testFile = 'test.csv'
        if os.path.exists(dataFile):
            os.remove(dataFile)
            print("Previous version of file removed!")
        df = self.frames['o3_daily']
        days = df['timestamp'].unique()
        print(days)
        
        train_MSE_list = np.empty((len(days),1))
        test_MSE_list = np.empty((len(days),1))

        # A fold for every day we have
        for i in range(1, len(days)):
            X_train, y_train, X_test, y_test = self.makeFeaturesForTesting(days[i-1], days[i], dataFile, testFile)
            # print('Test' + str(i))
            # print(X_train)
            reg = learning.regr(X_train, y_train)
            train_MSE = mean_squared_error(y_train, reg.predict(X_train))
            test_MSE = mean_squared_error(y_test, reg.predict(X_test))

            train_MSE_list[i,0] = train_MSE
            test_MSE_list[i,0] = test_MSE

        avg_train_scores = train_MSE_list.mean(axis=1)
        avg_test_scores = test_MSE_list.mean(axis=1)

        return avg_train_scores, avg_test_scores


    def makeFeaturesForTesting(self, start_day, end_day, data_file, test_file):
        # Will simply build testing features ontop of what we already have
        self.data_cleaner.gen_full_training_data(start_day, end_day, data_file)
        X_train, y_train = self.readCSV(data_file)
        
        start_day = end_day
        end_day += timedelta(days=1)
        if os.path.exists(test_file):
            os.remove(test_file)
            print("Previous version of test file removed!")
        self.data_cleaner.gen_full_training_data(start_day, end_day, test_file)
        X_test, y_test = self.readCSV(test_file)
        return X_train, y_train, X_test, y_test

    def readCSV(self, dataFile):
        # use pandas to read csv file
        df = pd.read_csv(dataFile, header=None)
        label_col = len(df.columns) - 1
        cols = [col for col in range(len(df.columns)) if col != label_col]
        X = df.iloc[:,cols].values
        # get label
        y = df.iloc[:,label_col].values
        print("Total dataset size: {} \n".format(len(X)))
        return X, y


