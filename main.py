#!/usr/bin/env python3
import argparse

import sys
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
            cleaner.consume_frame(df)
        return
    if args.subparser_name == "query":
        client = learning.EpaClient('query_storage')
        #sql = 'SELECT * FROM `{}.air_quality_annual_summary` LIMIT 10;'
        df = query_hawkins(client)
        print(df)
        #dataCleaner = learning.DataCleaner(df, 'query_storage')
        #dataCleaner.run()

        #arr = dataCleaner.toNumpyArray()
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
                time_local,
                parameter_name,
                latitude,
                longitude,
                sample_measurement,
                mdl,
                units_of_measure
            FROM
                `{}.nonoxnoy_hourly_summary`
            WHERE
                (state_code = "06" AND
				county_code = "073")
                ;
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
    return parser.parse_args()

if __name__ == "__main__":
    main()
