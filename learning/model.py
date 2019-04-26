import os

from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_squared_error

import learning

class Model:

    def __init__(self, trainingFile, regr):
        """Create a Model Object.

        Args:
            df: Data frame with all of the data
            cleaner: DataCleaner object used
        """
        self.arr = np.genfromtxt(trainingFile)
        print(self.arr)
        self.k_folds(regr=regr)

    def k_folds(self, regr = 'linear'):
        """Runs time based k-folds on a linear regression by default
            or some inputted classifier.

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
        days = self.arr[:,0].tolist()
        days = [datetime.utcfromtimestamp(x) for x in days]
        # Creating a test set which only includes data from 2017
        test_days = [day for day in days if self.forTraining(day)]
        # Finalizing the training set which includes data from before 2017
        days = [day for day in days if not self.forTraining(day)]

        train_MSE_list = np.empty((len(days),1))
        test_MSE_list = np.empty((len(days),1))

        X_test, y_test = self.makeFeaturesForTesting(test_days, 
            testFile)

        if regr == 'linear':
            reg = learning.linearregression.regr()
        elif regr == 'ridge':
            reg = learning.linearregression.ridge()
        elif regr == 'lasso':
            reg = learning.linearregression.lasso()
        else:
            reg = learning.linearregression.dummy()

        # A fold for every day we have
        for i in range(0, len(days), 4):
            # Adding training data for the day we are adding to the fold
            X_train, y_train = self.makeFeaturesForTraining(i, dataFile)
            reg.fit(X_train, y_train)
            train_MSE = mean_squared_error(y_train, reg.predict(X_train))
            test_MSE = mean_squared_error(y_test, reg.predict(X_test))

            train_MSE_list[i,0] = train_MSE
            test_MSE_list[i,0] = test_MSE

        avg_train_scores = train_MSE_list.mean(axis=0)
        avg_test_scores = test_MSE_list.mean(axis=0)

        print(min(train_MSE_list))
        print(min(test_MSE_list))
        print(avg_train_scores)
        print(avg_test_scores)
        return avg_train_scores, avg_test_scores


    def makeFeaturesForTraining(self, current_day, data_file):
        """Adds features for specified day into training file
            and makes training data set.

        Returns:
            The the training features and outputs
        """
        filehandle = open(data_file, 'a')
        np.savetxt(filehandle, self.arr[current_day:current_day+4], fmt='%.10e', delimiter = ',')
        filehandle.close()  # Make sure to close that file handle!
        X_train, y_train = self.readCSV(data_file)
        # Returning accumulated training data
        return X_train, y_train

    def makeFeaturesForTesting(self, test_days, test_file):
        """Making testing data set using the specified dates.

        Returns:
            The the testing features and outputs
        """
        # Remove previous version of test file
        if os.path.exists(test_file):
            os.remove(test_file)
            print("Previous version of test file removed!")
        filehandle = open(test_file, 'a')
        for index in range(len(self.arr)-len(test_days), len(self.arr)):
            # Save the output!
            np.savetxt(filehandle, self.arr[index:index+1], fmt='%.10e', delimiter = ',')
        filehandle.close()  # Make sure to close that file handle!
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
        cols = [col for col in range(len(df.columns)) if col != label_col and col!= 0]
        X = df.iloc[:,cols].values
        # get label
        y = df.iloc[:,label_col].values
        print("Total dataset size: {} \n".format(len(X)))
        return X, y

    def forTraining(self, dateTime):
        year = dateTime.year
        return year == 2011

    def getData(self, timeStamp, days):
        return

