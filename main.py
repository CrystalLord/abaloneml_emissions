#!/usr/bin/env python3
import argparse

import learning


def main():
    client = learning.EpaClient('query_storage')
    sql = 'SELECT y.latitude, y.longitude FROM (SELECT latitude, longitude, count(*) AS count_tuple FROM `{}.air_quality_annual_summary` GROUP BY latitude, longitude) y WHERE y.count_tuple = 8774;'
    df = client.query(sql)
    df.to_csv("testcsv")
    print(df)
    print("Successful!")


def parseargs():
    pass


if __name__ == "__main__":
    main()
