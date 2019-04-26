import os

from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_squared_error

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
        """Runs time based k-folds on a linear regression

        Returns:
            The average training MSE and average training MSE
        """
        dataFile = 'training.csv'
        testFile = 'test.csv'
        # Removing old version of the file training data will be read from
        if os.path.exists(dataFile):
            os.remove(dataFile)
            print("Previous version of file removed!")

        # Getting the dates for which we'll need training and testing data
        df = self.frames['o3_daily']
        days = df['timestamp'].unique()
        # Cutting out all of the days which aren't in the summer
        days = [day for day in days if self.isSummer(day)]
        # Creating a test set which only includes data from 2017
        test_days = [day for day in days if self.forTraining(day)]
        # Finalizing the training set which includes data from before 2017
        days = [day for day in days if not self.forTraining(day)]

        train_MSE_list = np.empty((len(days),1))
        test_MSE_list = np.empty((len(days),1))

        X_test, y_test = self.makeFeaturesForTesting(test_days[0], 
            test_days[len(test_days)-1]+  timedelta(days=1), 
            testFile)

        # A fold for every day we have
        for i in range(1, len(days)):
            # Adding training data for the day we are adding to the fold
            X_train, y_train = self.makeFeaturesForTraining(days[i], dataFile)
            reg = learning.regr(X_train, y_train)
            train_MSE = mean_squared_error(y_train, reg.predict(X_train))
            test_MSE = mean_squared_error(y_test, reg.predict(X_test))

            train_MSE_list[i,0] = train_MSE
            test_MSE_list[i,0] = test_MSE

        avg_train_scores = train_MSE_list.mean(axis=1)
        avg_test_scores = test_MSE_list.mean(axis=1)

        print(min(train_MSE_list))
        print(min(test_MSE_list))
        return avg_train_scores, avg_test_scores


    def makeFeaturesForTraining(self, current_day, data_file):
        """Adds features for specified day into training file
            and makes training data set.

        Returns:
            The the training features and outputs
        """
        end_day = current_day
        end_day += timedelta(days=1)
        # Will simply build testing features ontop of what we already have
        self.data_cleaner.gen_full_training_data(current_day, end_day, data_file)
        X_train, y_train = self.readCSV(data_file)
        # Returning accumulated training data
        return X_train, y_train

    def makeFeaturesForTesting(self, start_day, end_day, test_file):
        """Making testing data set using the specified dates.

        Returns:
            The the testing features and outputs
        """
        # Remove previous version of test file
        if os.path.exists(test_file):
            os.remove(test_file)
            print("Previous version of test file removed!")
        self.data_cleaner.gen_full_training_data(start_day, end_day, test_file)
        X_test, y_test = self.readCSV(test_file)
        return X_test, y_test

    def readCSV(self, dataFile):
        """ Reading a CSV of data and spliting it into features
            and outputs.

        Returns:
            The the features and outputs in the CSV
        """
        # use pandas to read csv file
        df = pd.read_csv(dataFile, header=None)
        label_col = len(df.columns) - 1
        cols = [col for col in range(len(df.columns)) if col != label_col]
        X = df.iloc[:,cols].values
        # get label
        y = df.iloc[:,label_col].values
        print("Total dataset size: {} \n".format(len(X)))
        return X, y

    def isSummer(self, dateTime):
        month = dateTime.month
        return month >= 6 and month <=9

    def forTraining(self, dateTime):
        year = dateTime.year
        return year == 2017

