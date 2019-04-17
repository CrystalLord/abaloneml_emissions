"""
Filename: epaclient.py
Author:
Date: 2019-04-05

Description:
    Holds the EpaClient class, which we may use to extract queries from the
    EPA dataset.

Example Usage:
    client = EpaClient('query_storage')
    sql = 'SELECT * FROM `{}.air_quality_annual_summary` LIMIT 10;'
    df = client.query(sql)  # Now in pandas dataframe.
"""


from google.cloud import bigquery
import os
import pandas as pd
import datetime


class EpaClient:

    def __init__(self,
                 storage_dir: str,
                 project='bigquery-public-data',
                 dsname='epa_historical_air_quality'):
        """Create an EPA Query Client.

        Args:
            storage_dir (str): name of directory to store query results.
            project (str): Project name. Default is 'bigquery-public-data'.
            dsname (str): Dataset name. Default is
                'epa_historical_air_quality'.
        """
        self.client = bigquery.Client()
        self.storage_dir = storage_dir
        self.project = project
        self.dsname = dsname
        self.location = "US"
        self.reset_job_config()

    def reset_job_config(self):
        """Resets the QueryJobConfig config."""
        self.job_config = bigquery.QueryJobConfig()
        # Use real runs. Consider setting to True when testing.
        self.job_config.dry_run = False
        # Allow looking into the query cache before getting the data from
        # Google.
        self.job_config.use_query_cache = True

    def query(self, sql: str, raw_sql=False, dry_run=False, save=True):
        """Runs a SQL query with the given job config on BigQuery.

        Args:
            sql (str): SQL call.
            raw_sql (bool): If the input sql is raw sql. Otherwise, the first
                {} is replaced with the dataset name. Default is 'False'.
            dry_run (bool): If set to True, does not conduct any API call.
                Default is 'False'.
        """
        if not raw_sql:
            query = sql.format(self.get_dataset_name())
        else:
            query = sql

        if dry_run:
            temp_dry_run_setting = self.job_config.dry_run
            self.job_config.dry_run = True

        print("Running the following query... (dry_run={})".format(dry_run))
        print("    > "+query)

        query_job = self.client.query(
            query,
            location=self.location,
            job_config=self.job_config
        )  # This line begins the query. This is an API call if wet.
        
        if dry_run:
            self.job_config.dry_run = temp_dry_run_setting
            return None
        print("Query completed. Converting to DataFrame.")
        df = query_job.to_dataframe()
        print("Conversion complete. Saving.")
        if save:
            self._mkdir_ifnotexist(self.storage_dir)
            now = datetime.datetime.today()
            df.to_csv(self.storage_dir + "/query_" + str(now))
        return df

    def get_dataset_name(self):
        """Returns the name of the dataset, in full."""
        return ".".join((self.project, self.dsname))

    def _mkdir_ifnotexist(self, dirpath):
        """Create a directory if it does not currently exist"""
        if os.path.isdir(dirpath):
            print(dirpath + " already exists. Skipping creation.")
        else:
            print("Creating directory " + dirpath + " ...")
            os.mkdir(dirpath)
