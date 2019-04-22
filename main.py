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

        cleaner.gen_full_training_data(datetime(2017, 6, 1),
                                       datetime(2017, 9, 1),
                                       'training_test.csv')
    if args.subparser_name == "query":
        client = learning.EpaClient('query_storage')
        #sql = 'SELECT * FROM `{}.air_quality_annual_summary` LIMIT 10;'
        df = query_hawkins(client)
        print(df)
        # dataCleaner = learning.DataCleaner(df, 'query_storage')
        #dataCleaner.run()

        #arr = dataCleaner.toNumpyArray()
    if args.subparser_name == "regr":
        arr = np.genfromtxt(args.datafile[0]);
        timestamps = arr[:,0].tolist()
        peak_ozone = arr[:,-1]
        peak_ozone = peak_ozone.reshape((len(peak_ozone), 1))

        X = arr[:,:-1]
        print(X)
        # TODO: change this to K-fold validation
        X_train, X_test, y_train, y_test = train_test_split(X, peak_ozone,
                                                            test_size=0.1)
        reg = learning.regr(X_train, y_train)

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
    regr_parser.add_argument('datafile', nargs=1, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    main()
