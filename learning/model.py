import os

from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
import sklearn.linear_model as linear_model
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit, RepeatedKFold
from sklearn.decomposition import PCA

import learning

"""
Start of the features in the numpy matrix. E.g. What parts aren't the
timestamp
"""
DATA_START_COL = 1
"""
Which column has the regression in it?
"""
REGR_COL = -1

class CrossValidator:

    def __init__(self, training_data, normalize=False, pca=None):
        """Create a Model Object.

        Args:
            df: Data frame with all of the data
            cleaner: DataCleaner object used
        """
        self.data = training_data
        if normalize:
            cut_out = self.data[:,DATA_START_COL:REGR_COL]
            cut_normed = preprocessing.normalize(cut_out)
            self.data[:,DATA_START_COL:REGR_COL] = cut_normed
        if pca is not None:
            decomp = PCA(n_components=pca)
            cut_out = self.data[:,DATA_START_COL:REGR_COL]
            cut_pca = decomp.fit_transform(cut_out)
            temp = np.concatenate((self.data[:,0].reshape(-1, 1), cut_pca,
                                   self.data[:,-1].reshape(-1, 1)), axis=1)
            self.data = temp
        self.models = []

    def k_folds(self, regr_name, fold_size=14, alpha_range=None):
        """Runs time based k-folds on a linear regression by default
            or some inputted classifier.

        Returns:
            The average training MSE and average training MSE
        """

        # specify the names of the two files we will use to create the k-folds.
        #dataFile = 'training.csv'
        #testFile = 'test.csv'
        # Removing old version of the file training data will be read from
        #if os.path.exists(dataFile):
        #    os.remove(dataFile)

        # Getting the dates for which we'll need training and testing data
        days = self.data[:, 0].tolist()
        days = [datetime.utcfromtimestamp(x) for x in days]

        # Get the number of folds in the dataset.
        num_training_folds = len(days)//fold_size - 1
        out_matrix = np.empty((num_training_folds-1, 5))

        # Loop through while the start of our fold is less
        # Make sure to start at 1, because we need to train on *something*.
        for fold_index in range(1, num_training_folds):

            # Find where our start and end indices are for the folds.
            train_fold_start = 0
            train_fold_end = fold_index * fold_size
            test_fold_end = (fold_index + 1) * fold_size

            # Slice our data for training and testing.
            X_train = self.data[train_fold_start:train_fold_end,
                                DATA_START_COL:REGR_COL]
            y_train = self.data[train_fold_start:train_fold_end,
                                REGR_COL].reshape(-1, 1)
            X_test = self.data[train_fold_end:test_fold_end,
                               DATA_START_COL:REGR_COL]
            y_test = self.data[train_fold_end:test_fold_end,
                               REGR_COL].reshape(-1, 1)

            # Train our regressor.
            out = self.nested_metatune(regr_name, X_train, y_train,
                                       alpha_range=alpha_range)
            alpha = out[0]
            train_mse = out[1]
            validation_mse = out[2]

            # Create a model with the best found alpha.
            regr = self.get_model(regr_name, alpha=alpha)
            regr.fit(X_train, y_train)
            self.models.append(regr)

            # Get the full train mse
            full_train_mse = mean_squared_error(y_train,
                                                regr.predict(X_train))
            test_mse = mean_squared_error(y_test, regr.predict(X_test))

            # Store the values in the numpy matrix.
            i = fold_index - 1
            out_matrix[i,0] = train_mse
            out_matrix[i,1] = validation_mse
            out_matrix[i,2] = full_train_mse
            out_matrix[i,3] = test_mse
            out_matrix[i,4] = 0 if (alpha is None) else alpha

        return out_matrix

        """
        # Creating a test set which only includes data from 2017
        test_days = [day for day in days if self.forTraining(day)]

        # Finalizing the training set which includes data from before 2017
        days = [day for day in days if not self.forTraining(day)]

        train_MSE_list = np.empty((len(days),1))
        test_MSE_list = np.empty((len(days),1))

        X_test, y_test = self.makeFeaturesForTesting(
            test_days,
            testFile
        )

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
        """
        return train_mse_arr, test_mse_arr
        #return avg_train_scores, avg_test_scores

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
        filehandle = open(test_file, 'a')
        for index in range(len(self.arr)-len(test_days), len(self.arr)):
            # Save the output!
            np.savetxt(filehandle, self.arr[index:index+1], fmt='%.10e', delimiter = ',')
        filehandle.close()  # Make sure to close that file handle!
        X_test, y_test = self.readCSV(test_file)
        return X_test, y_test

    def read_csv(self, datafile):
        """Reads a CSV of data and puts it into a numpy matrix.

        Args:
            datafile (str): CSV file to read in.

        Returns:
            numpy matrix formed from the CSV file.
        """
        data = np.genfromtxt(datafile, delimiter=',')
        return data

    def nested_metatune(self, regr_name, foldX, foldY, alpha_range=None,
                        validation_frac=0.5):
        """Conducts the nested metaparameter tuning for time-based cross
        validation.

        That is, given a fold's features and a fold's regression values,
        generate a model for each setting in the alpha_range, and test each on
        a train/validation split according to the validation_frac above.

        Args:
            regr_name (str): String of the regressor's name. Can be 'mean',
                'lasso', or 'ridge'.
            foldX (np.ndarray): Fold's feature parameter matrix. Excludes
                timestamps.
            foldy (np.ndarray): Fold's regression variable array.
            alpha_range (list, optional): List of alphas we may want to test.
            validation_frac (float, optional):

        Returns:
            A tuple of the best alpha (None if not applicable), training error,
            and test error in that order.
        """
        # Split the data.
        X_train, X_vald, y_train, y_vald = self.split_validation(
            foldX,
            foldY,
            validation_frac=validation_frac
        )

        # Check if we're doing a dummy classifier.
        # Or just any classifier without metaparams.
        if alpha_range is None:
            regr = self.get_model(regr_name)
            regr.fit(X_train, y_train)
            train_err = mean_squared_error(y_train, regr.predict(X_train))
            vald_err = mean_squared_error(y_vald, regr.predict(X_vald))
            return None, train_err, vald_err

        # Loop through to find best alphas.
        best_alpha = None
        train_err = None
        vald_err = None
        for alpha in alpha_range:
            regr = self.get_model(regr_name, alpha=alpha)
            regr.fit(X_train, y_train)
            # _h for hypothesis
            train_err_h = mean_squared_error(y_train, regr.predict(X_train))
            vald_err_h = mean_squared_error(y_vald, regr.predict(X_vald))
            if best_alpha is None:
                best_alpha = alpha
                train_err = train_err_h
                vald_err = vald_err_h
            elif vald_err_h < vald_err:
                best_alpha = alpha
                train_err = train_err_h
                vald_err = vald_err_h

        # Return the best alpha, and its respective train and validation err
        return best_alpha, train_err, vald_err


    def split_validation(self, x, *args, validation_frac=0.2):
        """Splits the provided numpy array into the training and validation,
        where the validation is always the latter fraction of the data.

        Args:
            x: Matrix to split.
            *args: Additional matrices to split.
            validation_frac: Percent of data to use for
        """
        n, _ = x.shape
        split_index = int(round((n-1) * (1-validation_frac)))
        if split_index == 0:
            raise ValueError("Invalid validation fraction: no training data")

        # Split the data
        x_train = x[:split_index,:]
        x_vald = x[split_index:,:]
        output_list = [x_train, x_vald]
        for a in args:
            try:
                a_train = a[:split_index,:]
                a_vald = a[split_index:,:]
            except IndexError:
                a_train = a[:split_index]
                a_vald = a[split_index:]
            output_list += [a_train, a_vald]
        return output_list

    def simple_k_folds(self, regr_name, alpha_range=None,
                       fold_size=14, num_test=93):
        """Simple CV.
        """
        days = self.data[:, 0].tolist()
        days = [datetime.utcfromtimestamp(x) for x in days]

        train_data, test_data = self.cut_on_date(datetime(2011, 1, 1),
                                                 self.data)

        # Get the number of folds in the dataset.
        num_training_folds = (len(train_data)-num_test)//fold_size - 1
        out_matrix = np.empty((num_training_folds-1, 5))

        # Loop through while the start of our fold is less
        # Make sure to start at 1, because we need to train on *something*.
        for fold_index in range(1, num_training_folds):

            # Find where our start and end indices are for the folds.
            train_fold_start = 0
            train_fold_end = fold_index * fold_size
            test_fold_end = (fold_index + 1) * fold_size

            # Slice our data for training and testing.
            X_train = self.data[train_fold_start:train_fold_end,
                                DATA_START_COL:REGR_COL]
            y_train = self.data[train_fold_start:train_fold_end,
                                REGR_COL].reshape(-1, 1)
            X_vald = self.data[train_fold_end:test_fold_end,
                               DATA_START_COL:REGR_COL]
            y_vald = self.data[train_fold_end:test_fold_end,
                               REGR_COL].reshape(-1, 1)

            regr = self.get_model(regr_name)
            regr.fit(X_train, y_train)
            self.models.append(regr)

            full_train_mse = mean_squared_error(y_train,
                                                regr.predict(X_train))
            vald_mse = mean_squared_error(y_vald, regr.predict(X_vald))

            i = fold_index - 1
            out_matrix[i,0] = 0
            out_matrix[i,1] = vald_mse
            out_matrix[i,2] = full_train_mse
            out_matrix[i,3] = 0
            out_matrix[i,4] = 0

        # Slice our data for training and testing.
        final_train_mse, final_test_mse = self.final_test(
            regr_name,
            train_data,
            test_data
        )

        for i in range(num_training_folds-1):
            out_matrix[i, 3] = final_test_mse
        return out_matrix

    def get_model(self, regr_name, **kwargs):
        """Returns a scikit learn model mapped to the provided name. Keyword
            args are put into the model creator.
        """
        if regr_name == "lasso":
            return linear_model.Lasso(max_iter=5000, **kwargs)
        elif regr_name == "ridge":
            return linear_model.Ridge(max_iter=5000, **kwargs)
        elif regr_name == "linear":
            return linear_model.LinearRegression()
        elif regr_name == "mean":
            return DummyRegressor('mean')
        else:
            raise ValueError(f"Unrecognised model name {regr_name}")

    def cut_on_date(self, cut_date, x, *args):
        days = x[:, 0].tolist()
        days = [datetime.utcfromtimestamp(x) for x in days]

        split_index = x.shape[0]
        for ind, d in enumerate(days):
            if d >= cut_date:
                split_index = ind
                break

        x_train = x[:split_index,:]
        x_vald = x[split_index:,:]
        output_list = [x_train, x_vald]
        for a in args:
            try:
                a_train = a[:split_index,:]
                a_vald = a[split_index:,:]
            except IndexError:
                a_train = a[:split_index]
                a_vald = a[split_index:]
            output_list += [a_train, a_vald]
        return output_list

    def final_test(self, regr_name, train_data, test_data,
                   model_params={}):
        """Conducts a final test on held out data"""
        X_train = train_data[:,
                            DATA_START_COL:REGR_COL]
        y_train = train_data[:,
                            REGR_COL].reshape(-1, 1)
        X_test = test_data[:,
                           DATA_START_COL:REGR_COL]
        y_test = test_data[:,
                           REGR_COL].reshape(-1, 1)

        # Generate the model we want with the desired params.
        regr = self.get_model(regr_name, **model_params)
        regr.fit(X_train, y_train)
        train_mse = mean_squared_error(y_train, regr.predict(X_train))
        test_mse = mean_squared_error(y_test, regr.predict(X_test))


        self.test_true = y_test
        self.test_pred = regr.predict(X_test)

        return train_mse, test_mse

    def super_simple_cv(self, regr_name, alpha_range=None,
                       fold_num=6):
        """Simple CV.
        """

        splitter = ShuffleSplit(n_splits=1, test_size=.10,
                                random_state=2)
        # This for loop only gets run once.
        train_index, test_index = list(splitter.split(self.data))[0]
        X_train = self.data[train_index,
                            DATA_START_COL:REGR_COL]
        y_train = self.data[train_index,
                            REGR_COL].reshape(-1, 1)
        X_test = self.data[test_index,
                            DATA_START_COL:REGR_COL]
        y_test = self.data[test_index,
                            REGR_COL].reshape(-1, 1)

        if alpha_range is not None:
            best_alpha, vald_mse = self.repeat_k_folds(X_train,
                                                       y_train,
                                                       regr_name,
                                                       alpha_range,
                                                       fold_num)
        else:
            best_alpha = 0
            vald_mse = 0
        # Generate a new model with the best alpha we found.
        regr = self.get_model(regr_name, alpha=best_alpha)
        regr.fit(X_train, y_train)
        self.models.append(regr)

        full_train_mse = mean_squared_error(y_train,
                                            regr.predict(X_train))
        test_mse = mean_squared_error(y_test, regr.predict(X_test))

        self.test_true = y_test
        self.test_pred = regr.predict(X_test)

        # Make an output matrix for easy use.
        out_matrix = np.empty((1, 5))
        out_matrix[0,0] = 0
        out_matrix[0,1] = vald_mse
        out_matrix[0,2] = full_train_mse
        out_matrix[0,3] = test_mse
        out_matrix[0,4] = best_alpha

        return out_matrix

    def repeat_k_folds(self,
                       X,
                       y,
                       regr_name,
                       alpha_range,
                       num_splits,
                       num_times=5):
        """Conducts repeated k-Folds on the data, for the provided number of
            times
        """
        best_alpha = None
        best_err = float("inf")
        for alpha in alpha_range:
            # Get the first alpha around.
            if best_alpha is None:
                best_alpha = alpha

            # Keep a running sum of errors over repeated k-folds
            err_sum = 0
            count = 0
            # Split for K-Folds.
            folder = RepeatedKFold(n_splits=num_splits,
                                  n_repeats=num_times,
                                  random_state=1234)
            regr = self.get_model(regr_name, alpha=alpha)
            for train_ind, validate_ind in folder.split(X, y):
                # Slice training data
                X_train = X[train_ind, :]
                y_train = y[train_ind, :]
                X_vald = X[validate_ind, :]
                y_vald = y[validate_ind, :]

                # Update error and alpha
                regr.fit(X_train, y_train)
                err_sum += mean_squared_error(y_vald, regr.predict(X_vald))
                count += 1
            avg_err = err_sum/count
            print(f"alph: {alpha} err: {avg_err}")
            if avg_err < best_err:
                best_err = avg_err
                best_alpha = alpha
        print("K-folds found...")
        print(f"    best_alpha = {best_alpha}")
        print(f"    best_err = {best_err}")
        return best_alpha, best_err
