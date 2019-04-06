#!/usr/bin/env python3
import argparse

import learning


def main():
    client = learning.EpaClient('query_storage')
    sql = 'SELECT * FROM `{}.air_quality_annual_summary` LIMIT 10;'
    df = client.query(sql)
    print(df)
    print("Successful!")


def parseargs():
    pass


if __name__ == "__main__":
    main()
