#!/usr/bin/env python3
import argparse

import learning


def main():
    client = learning.EpaClient('query_storage')
    #sql = 'SELECT * FROM `{}.air_quality_annual_summary` LIMIT 10;'
    #df = client.query(sql, dry_run=True)
    df = query_hawkins(client)
    print(df)
    dataCleaner = learning.DataCleaner(df)
    dataCleaner.printColumnValue()
    print("Successful!")


def query_hawkins(client):
    sql = '''SELECT
                (state_code,
                county_code,
                site_num,
                date_local,
                time_local,
                parameter_name,
                latitude, longitude,
                sample_measurement,
                mdl,
                units_of_measure)
            FROM
                `{}.o3_hourly_summary`
            WHERE
                (state_code = "06" AND
                county_code = "065" AND
                site_num = "8001");
            '''
    df = client.query(sql, dry_run=False)
    return df


def parseargs():
    pass


if __name__ == "__main__":
    main()
