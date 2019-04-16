from google.cloud import bigquery
import os
import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataCleaner:

    def __init__(self, storage_dir):
        """Create a DataCleaner Object.

        Args:
            df: Data frame with all of the data
        """
        self.storage_dir = storage_dir
        self.frames = {}

    def consume_frame(self, df, frame_name=None):
        if frame_name is None:
            num_frames = len(self.frames.keys())
            frame_name = "frame_" + str(num_frames)
        # Add the frame
        self.frames[frame_name] = df
        # Append the hour numerically.
        # DEPRECATED
        #self.append_numeric_hour(frame_name)
        # Append the timestamp column
        self.append_full_timestamp(frame_name)
        # Sort by time.
        self.frames[frame_name].sort_values('timestamp', inplace=True)
        # Reset the indices after sorting, and drop the old index.
        self.frames[frame_name] = self.frames[frame_name].reset_index(drop=True)

#    def run(self, frame_name):
#        self.drop_columns()
#        #self.print_column_value()
#        nparray = self.to_numpy_array()
#        print(nparray)
#        self.sanity_check(nparray)
#        np.save(self.storage_dir + "/query_" +
#                str(datetime.datetime.today()), nparray)
#        print("Successful!")

    def to_numpy_array(self, frame_name):
        return self.frames[frame_name].to_numpy()

    #def print_column_value(self):
    #    print(self.frames.columns.tolist())

    def generate_npdata(self):
        """Produces the training and test data together as one long numpy
        matrix

        Returns:
            A tuple of the Numpy ndarrays where the first element is the
            training data, and the second element is the regression data.
        """
        X = np.empty((0,0))
        for fr in self.frames.keys():
            col = self.frames[fr][["sampled_measurement"]]\
                .to_numpy(dtype=np.float64)
        return X, y

    def append_full_timestamp(self, frame_name):
        """Appends the full timestamp object to the dataframe"""
        df = self.frames[frame_name]
        dateseries = df['date_local']
        hourseries = df['numeric_hour']
        new_series = pd.Series(np.empty(dateseries.size,))
        for i, t in dateseries.iteritems():
            oldtime = t.split("-")
            year = int(oldtime[0])
            month = int(oldtime[1])
            day = int(oldtime[2])
            hour = int(hourseries[i])
            new_series[i] = datetime.datetime(year,  # Year
                                              month,  # Month
                                              day,  # Day
                                              hour,  # Hour
                                              0  # Minute
            )
        self.frames[frame_name]['timestamp'] = new_series

    def data_at_time(self, frame_name, year, month, day, hour):
        """Returns a data frame containing instances of samples from each
        stored sampled_measurement column at a particular hour

        Args:
            frame_name (str):
            year (int):
            month (int):
            day (int):
            hour (int):

        Returns:
            Returns the full row from the dataframe which matches this date.
            If multiple rows match this time, it will select the row
            arbitrarily.
        """
        df = self.frames[frame_name]
        # Get the date time object to compare with.
        t = datetime.datetime(year, month, day, hour)

        # set up our binary search.
        current_index = len(df)//2
        max_bound = len(df)-1
        min_bound = 0
        while True:
            comparison_time = df.loc[current_index, 'timestamp']
            if t == comparison_time:
                print("found!")
                break
            elif max_bound == min_bound:
                print("Not found")
                break
            else:
                if t < comparison_time:
                    # Our time is earlier!
                    max_bound = current_index
                    current_index = (max_bound + min_bound)//2
                else:
                    # Our time is later!
                    min_bound = current_index + 1
                    current_index = (min_bound + max_bound)//2


    def append_numeric_hour(self, frame_name=None):
        """Appends the "numeric_hour" column to each data frame. This column
            stores the hour of the sample as a floating point number.
        """
        if frame_name is None:
            for fr in self.frames.keys():
                # Iterate through the local_time values, and convert them to
                # floats.
                series = self.frames[fr]['time_local']
                new_series = pd.Series(np.empty((series.size,)),
                                       name='numeric_hour')
                for i, s in series.iteritems():
                    hr, mn = s.split(":")
                    new_series[i] = float(hr)
                self.frames[fr].append(new_series)
                self.frames[fr]['numeric_hour'] = new_series
        else:
            series = self.frames[frame_name]['time_local']
            new_series = pd.Series(np.empty((series.size,)),
                                   name='numeric_hour')
            for i, s in series.iteritems():
                hr, mn = s.split(":")
                new_series[i] = float(hr)
            self.frames[frame_name]['numeric_hour'] = new_series


    def drop_columns(self, frame_name=None):
        """Drops unnecessary columns from dataframes.

        Args:
            frame_name (str, optional): Name of frame to drop columns from. If
                not specified, drops columns from all data frames consumed.
        Post:
            Columns are dropped from the internal data frames.
        """
        #'state_code', 'county_code', 'site_num', 'date_local', 'time_local',
        #'parameter_name', 'latitude', 'longitude', 'sample_measurement', 'mdl',
        # 'units_of_measure']
        if frame_name is not None:
            # If a frame is specified, drop these columns only for that frame.
            self.frames[frame_name] = self.frames[frame_name].drop(
                ['state_code',
                 'county_code',
                 'site_num',
                 'latitude',
                 'longitude',
                 'mdl'],
                axis=1
            )
        else:
            # If a frame is not specified, drop these columns from all frames.
            for fr in self.frames.keys():
                self.frames[fr] = self.frames[fr].drop(
                    ['state_code',
                     'county_code',
                     'site_num',
                     'latitude',
                     'longitude',
                     'mdl'],
                    axis=1
                )

    def one_hot(self, col_name):
        """Produces a one-hot numpy matrix from all the entries in the produced
            column name.

        Args:
            col_name (str): Name of the column in all dataframes which need to
                be converted into one-hot.
        Returns:
            A numpy binary matrix with shape (num_entries, num_categories).
            Represents a one-hot encoding.
        """
        enc = OneHotEncoder(handle_unknown='error')
        values = []
        for fr in self.frames.keys():
            df = self.frames[fr]
            values += df['parameter_name'].tolist()
        # Convert to numpy array because scikit learn needs things in a 2D
        # array format.
        values_np = np.asarray(values).reshape(len(values), 1)
        onehot_values = enc.fit_transform(values_np).toarray()
        return onehot_values

    def uniform_ppm(self,
                    sample_name="sample_measurement",
                    unit_name="units_of_measure"):
        """Converts all sample values to PPM from PPB or PPT"""
        for index, row in df.iterrows():
            if 'billion' in row[unit_name].tolower():
                row[sample_name] /= 1000.0
            if 'trillion' in row[unit_name].tolower():
                row[sample_name] /= 10000000.0

    def sanity_check(self, array_check):
        print(np.unique(array_check[:, 0]))
        print(np.unique(array_check[:, 4]))


"""
time stamp
sample_measurement
sanity check: parameter_name all same
sanity check: all same measurements unit
multiplicative factor between different units
one_hot for parameter_name
Ozone
"""
