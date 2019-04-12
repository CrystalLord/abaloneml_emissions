from google.cloud import bigquery
import os
import pandas as pd
import datetime
import numpy as np


class DataCleaner:

    def __init__(self, df, storage_dir):
        """Create a DataCleaner Object.

        Args:
            df: Data frame with all of the data
        """
        self.df = df
        self.storage_dir = storage_dir

    def run(self):
        self.dropColumns()
        self.printColumnValue()
        npArray = self.toNumpyArray()
        print(npArray)
        self.sanityCheck(npArray)
        np.save(self.storage_dir + "/query_" + str(datetime.datetime.today()), npArray)
        print("Successful!")

    def toNumpyArray(self):
        return self.df.to_numpy()

    def printColumnValue(self):
        print(self.df.columns.tolist())

    def dropColumns(self):
        #'state_code', 'county_code', 'site_num', 'date_local', 'time_local', 
        #'parameter_name', 'latitude', 'longitude', 'sample_measurement', 'mdl', 
        #'units_of_measure']
        self.df = self.df.drop(['state_code', 'county_code', 'site_num', 'latitude', 'longitude', 'mdl'], axis = 1)

    def sanityCheck(self, arrayCheck):
        print(np.unique(arrayCheck[:,2]))
        print(np.unique(arrayCheck[:,4]))

"""
time stamp
sample_measurement
sanity check: parameter_name all same
sanity check: all same measurements unit
multiplicative factor between different units
one_hot for parameter_name
Ozone
"""