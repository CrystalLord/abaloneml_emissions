from google.cloud import bigquery
import os
import pandas as pd
import datetime


class DataCleaner:

    def __init__(self, df):
        """Create a DataCleaner Object.

        Args:
            df: Data frame with all of the data
        """
        self.df = df

    def toNumpyArray(self):
        return 

    def printColumnValue(self):
        print(self.df.columns.tolist())

    def updateColumn(self, column):
        return 

"""
time stamp
sample_measurement
sanity check: parameter_name all same
sanity check: all same measurements unit
multiplicative factor between different units
one_hot for parameter_name
Ozone
"""