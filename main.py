#!/usr/bin/env python3
import argparse

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
#from sklearn.linear_regression import LinearRegression

import learning
import plotting


DESCRIPTION = (
    """Historical Air Quality Dataset Analyser."""
)


def main():
    args = parseargs()

    if args.subparser_name == 'ml':
        cleaner = learning.DataCleaner('query_storage')
        for fp in args.filenames:
            df = pd.read_csv(fp)
            # name = fp.split("/")[-1]
            measurement, time_type = fp.split("_")[-2:]
            time_type = time_type.split(".")[0]
            name = measurement + "_" + time_type

            split_on_param = (measurement != "voc")
            if time_type == "daily":
                cleaner.consume_frame(df, "daily", frame_name=name,
                                      split_params=split_on_param)
            if time_type == "hourly":
                cleaner.consume_frame(df, "hourly", frame_name=name,
                                      split_params=split_on_param)


        # cleaner.gen_full_training_data(datetime(2017, 9, 1),
        #                                datetime(2017, 10, 1),
        #                                'training_test.csv')
        for year in range(2002,2012):
            fp = 'training_data_summer_{}'.format(year)
            cleaner.gen_full_training_data(datetime(year, 6, 1),
                                           datetime(year, 9, 1),
                                           fp)
    if args.subparser_name == "query":
        client = learning.EpaClient('query_storage')
        #sql = 'SELECT * FROM `{}.air_quality_annual_summary` LIMIT 10;'
        df = query_hawkins(client)
        print(df)

    if args.subparser_name == "regr":
        trainingFile = args.datafile
        # model = learning.Model(trainingFile) # Is this necessary?

        # Extract the dataset. To be replaced with CV by Pryianka.
        # -----------------------------------------------------------------
        regr_name = args.regressor
        normalize = True

        data = np.genfromtxt(args.datafile)
        weekenddata = data[:, [0, -8, -7, -6, -5, -4, -3, -2, -1]]

        validator = learning.CrossValidator(weekenddata,
                                            normalize=normalize)

        out_matrix = validator.super_simple_k_folds(
            regr_name,
            alpha_range=[1.0]
        )

        #print(validator.models[-1].coef_)
        #print(validator.models[-1].intercept_)
        header = "train_mse,validation_mse,full_train_mse,test_mse,alphas"
        np.savetxt(f"{regr_name}_cv_output_mat.csv",
                   out_matrix,
                   delimiter=',',
                   header=header)

        train_mse = out_matrix[:,0]
        validation_mse = out_matrix[:,1]
        full_train_mse = out_matrix[:,2]
        test_mse = out_matrix[:,3]
        alphas = out_matrix[:,4]

        print(f"Full Train Mean MSE {full_train_mse.mean()}")
        print(f"Test Mean MSE {test_mse.mean()}")

        # Set up title.
        title = f"'{regr_name}' Regression with Time Nested CV"
        if normalize:
            title += " (Normalized)"

        plt.plot(full_train_mse*100, label='Full Train MSE')
        plt.plot(test_mse*100, label='Test MSE')
        plt.title(title)
        plt.ylabel("% Error & Alpha")
        plt.xlabel('Week (Validation Fold)')
        plt.legend()
        plt.savefig(f"{regr_name}_cv_plot.png", dpi=196)
        plt.show()

        # Show alphas.
        plt.plot(alphas, '--', label='Alpha')
        plt.show()

    if args.subparser_name == 'meta':
        # Scan through metaparameters
        for alpha in (0, 1, 10, 100, 1000, 10**4, 10**5):
            ridge_regr = learning.ridge(alpha=alpha)
            lasso_regr = learning.lasso(alpha=alpha)

        # X_train, X_test, y_train, y_test = train_test_split(X, peak_ozone, test_size=0.1)
        # reg = learning.model(X_train, y_train)

        # predictions = reg.predict(X)
        test_predictions = reg.predict(X_test)
        test_times = timestamps[-len(X_test):]

        print("Num training: {}".format(X_train.shape))
        print("Num test: {}".format(X_test.shape))
        print("Coefs:", np.argmax(reg.coef_))
        print("Train MSE:",mean_squared_error(y_train, reg.predict(X_train)))
        print("Test MSE:",mean_squared_error(y_test, reg.predict(X_test)))

        plt.scatter(test_times, test_predictions, s=20)
        plt.vlines(test_times, test_predictions, y_test)
        plt.scatter(test_times, y_test, s=20)
        plt.title('Linear Regression')
        plt.xlabel('Time')
        plt.ylabel('Peak Ozone, Parts Per Million')
        plt.show()

    if False:
        # This is what I used to produce the baseline.
        arr = np.load('query_SD_o3.npy');
        hourlist = arr[:,1].tolist()
        hourlist = [float(x[:2]) for x in hourlist]
        ozone = arr[:,3]
        ozone = ozone.reshape((len(ozone), 1))

        hourlistnp = np.asarray(hourlist).reshape(len(hourlist), 1)

        poly = PolynomialFeatures(degree=4)
        X = poly.fit_transform(hourlistnp)

        X_train, X_test, y_train, y_test = train_test_split(X, ozone,
                                                            test_size=0.1)

        reg = learning.regr(X, ozone)
        predictions = reg.predict(X)

        print("Num training: {}".format(X_train.shape))
        print("Num test: {}".format(X_test.shape))
        print("Coefs:", reg.coef_)
        print("Train MSE:",mean_squared_error(y_train, reg.predict(X_train)))
        print("Test MSE:",mean_squared_error(y_test, reg.predict(X_test)))

        plt.scatter(hourlistnp, ozone, alpha=0.1, s=0.5)
        plt.scatter(hourlistnp, predictions)
        plt.title('4th Order Linear Regression')
        plt.xlabel('Hour of Day, Local Time')
        plt.ylabel('Ozone, Parts Per Million')
        plt.show()


def query_hawkins(client):
    sql = '''SELECT
                state_code,
                county_code,
                site_num,
                date_local,
                parameter_name,
                latitude,
                longitude,
                arithmetic_mean,
                first_max_value,
                first_max_hour,
                units_of_measure
            FROM
                `{}.no2_daily_summary`
            WHERE
                state_code = '06' AND county_code = '073'
            '''
    df = client.query(sql, dry_run=False)
    return df

def parseargs():
    """Parse user arguments."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    subparsers = parser.add_subparsers(help='Operations you can do...',
                                       dest='subparser_name')
    subparsers.required = True
    query_parser = subparsers.add_parser('query',
                                         help='Conduct query operations')
    ml_parser = subparsers.add_parser(
        'ml', help='Conduct machine learning operations')
    ml_parser.add_argument('filenames', nargs='+', type=str)

    regr_parser= subparsers.add_parser(
        'regr', help='Train regressor and view predictions')
    regr_parser.add_argument('datafile', type=str)
    regr_parser.add_argument('regressor', type=str,
                             default='linear',
                             help="Can be 'linear', 'ridge', 'lasso', or 'mean'")
    return parser.parse_args()


def mean_of_week(data):
    swend = 0
    swday = 0
    cwend = 0
    cwday = 0
    for i in range(data.shape[0]):
        if data[i, -2] == 1.0 or data[i, -3] == 1.0:
            # Is weekend.
            swend += data[i,-1]
            cwend += 1
        else:
            swday += data[i,-1]
            cwday += 1
    return swday/cwday, swend/cwend

if __name__ == "__main__":
    main()
