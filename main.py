#!/usr/bin/env python3
import argparse

import learning
import plotting


def main():
    client = learning.EpaClient('query_storage')
    # sql = 'SELECT * FROM `{}.air_quality_annual_summary` LIMIT 10;'
    # df = client.query(sql, dry_run=False)
    df = query_san_diego(client)
    print(df)
    # dataCleaner = learning.DataCleaner(df, 'query_storage')
    # dataCleaner.run()
    
def query_san_diego(client):
    sql = '''SELECT 
                state_code,
                county_code,
                site_num,
                count(*) tuple
            FROM
                `{}.air_quality_annual_summary`
            WHERE
                state_code = '06' AND county_code = '073' AND site_num = '1001'
            GROUP BY 
                state_code, county_code, site_num
    '''
    df = client.query(sql, dry_run=False)
    return df 



def parseargs():
    pass


if __name__ == "__main__":
    main()
