import os

from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta
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

    def consume_frame(self, df, frame_type, frame_name=None):
        # check how many unique parameter names there are
        param_names = pd.Series.unique(df['parameter_name'])
        dataframes = []
        if len(param_names) == 1:
            dataframes = [df]
        else:
            for p in param_names:
                dftemp = df.loc[df['parameter_name'] == p]
                dataframes.append(dftemp)  
        for d in dataframes:
            print("Consuming dataframes")
            if frame_name is None:
                num_frames = len(self.frames.keys())
                frame_name = "frame_" + str(num_frames)
            # Add the frame
            self.frames[frame_name] = d
            # Append the hour numerically.
            # DEPRECATED
            #self.append_numeric_hour(frame_name)
            if frame_type == 'hourly':
                # Append the timestamp column for hourly df
                self.append_full_timestamp(frame_name)
            if frame_type == 'daily':
                self.frames[frame_name]['timestamp'] = d['date_local']
            # Sort by time.
            self.frames[frame_name].sort_values('timestamp', inplace=True)
            # Reset the indices after sorting, and drop the old index.
            self.frames[frame_name] = self.frames[frame_name].reset_index(
                drop=True
            )
            # get either hourly features for a given day or average features for previous day
            if frame_type == 'hourly':
                print(self.gen_day_features([frame_name],
                                        datetime(2010, 6, 4, 12), 4, 3))
            elif frame_type == 'daily':
                print("in gen_avg_day_features")
                print(self.gen_day_avg_features([frame_name],
                                        datetime(2010, 6, 4, 12), 1))
            # else # I think the remaining case would be for previous year

#    def run(self, frame_name):
#        self.drop_columns()
#        #self.print_column_value()
#        nparray = self.to_numpy_array()
#        print(nparray)
#        self.sanity_check(nparray)
#        np.save(self.storage_dir + "/query_" +
#                str(datetime.today()), nparray)
#        print("Successful!")

    def to_numpy_array(self, frame_name):
        return self.frames[frame_name].to_numpy()

    def gen_full_training_data(self, day_look_back, year_look_back):
        """Produces the training and test data together as one long numpy
        matrix

        Returns:
            A tuple of the Numpy ndarrays where the first element is the
            training data, and the second element is the regression data.
        """
        raise NotImplemented("Working on it!")

    def gen_day_features(self, frames_of_interest, day_of_interest,
                         day_look_back, year_look_back, hour_range=(0,24)):
        """Generates features for a day of interest

        Args:
            frame_of_interest (list(str)): Name of the DataFrames to look at.
            day_of_interest (datetime): Day to calculate features from.
            day_look_back (int): Days prior to the one of interest to extract
                data from.
            year_look_back (int): Years to get data from on each day we are
                looking back on.
            hour_range (tuple, optional): Range of data

        Returns:
            Returns a numpy ndarray of dimensions:
                (1, (day_look_back*year_look_back - 1)*num_frames*num_hours).
                To understand what this means, we return the sampled data at
                every hour for every frame for every day we look back for every
                year we look back. However, we do not include the day of
                interest, therefore we remove 1 day of all years.
        """
        num_frames = len(frames_of_interest)
        num_hours = hour_range[1] - hour_range[0]
        # Create an empty feature matrix first, then fill it up.

        features = np.empty(
            (1, (day_look_back*year_look_back - 1)*num_frames*(num_hours)) 
        )

        # Keep track of which feature we are writing to with this counter.
        counter = 0
        for frame in frames_of_interest:
            for year in range(year_look_back):
                for day in range(day_look_back):
                    if year == 0 and day == 0:
                        continue
                    # Loop through all hours we care about.
                    last_hour_index = None
                    for hour in range(hour_range[0], hour_range[1]):
                        # Compute the delta time, and find the time when we want to
                        # grab the sample.
                        true_year = int(round((year*365.25)))
                        delta = timedelta(days=-(day+true_year), hours=-hour)

                        sample_time = day_of_interest + delta
                        if last_hour_index is None:
                            s, ind = self.sample_at_time(frame, sample_time,
                                                    return_nearest=True)
                            last_hour_index = ind
                        else:
                            hi_sample = min(last_hour_index + 100,
                                len(self.frames[frame]))
                            lo_sample = max(last_hour_index - 100, 0)
                            s, ind = self.sample_at_time(
                                frame,
                                sample_time,
                                return_nearest=True,
                                lower_bound=lo_sample,
                                upper_bound=hi_sample
                            )
                            last_hour_index = ind
                        features[0, counter] = s
                        # Make sure to increment the counter to write to the
                        # next feature cell!
                        counter += 1
                            
        return features

    def gen_day_avg_features(self, frames_of_interest, day_of_interest,
                         day_look_back):
        """Generates features for the <day_look_back> days before the day of interest

        Args:
            day_of_interest (datetime):
        """
        num_frames = len(frames_of_interest)
        # num_hours = hour_range[1] - hour_range[0]
        # Create an empty feature matrix first, then fill it up.

        features = np.empty(
            (1, (day_look_back*2)) 
        )

        # Keep track of which feature we are writing to with this counter.
        counter = 0
        for day in range(day_look_back):
            if day == 0:
                continue
            # Compute the delta time, and find the time when we want to
            # grab the sample.
            delta = timedelta(days=-day)
            sample_time = day_of_interest + delta
            
            sample_time.hour = 0
            sample_time.minute = 0
            sample_time.second = 0

            for frame in frames_of_interest:
                s_max, s_avg = self.sample_at_day(frame, sample_time,
                                        return_nearest=True)
                
                features[0, counter] = s_max
                counter += 1
                features[0, counter] = s_avg
                counter += 1
        return features

    def mean_between_dates(self, frame_name, startdate, enddate,
                           colname='sample_measurement'):
        """Calculates the mean frame value between to dates (or the closest
        values of those dates).

        Args:
            date1 (datetime): Start date, inclusive.
            date2 (datetime): End date, exclusive.
            colname (str, optional): Name of column we want the mean across.
                Default is 'sample_measurement'.

        Returns:
            Mean (float) of the selected column.
        """
        _, ind1 = self.data_at_time(frame_name, startdate, return_nearest=True)
        _, ind2 = self.data_at_time(frame_name, enddate, return_nearest=True)
        df = self.frames[frame_name]
        return df.loc[ind1:ind2, colname].mean(axis=0)

    def append_full_timestamp(self, frame_name):
        """Appends the full timestamp object to the dataframe"""
        df = self.frames[frame_name]
        dateseries = df['date_local']
        hourseries = df['time_local']
        new_series = pd.Series(np.empty(dateseries.size,))
        for i, t in dateseries.iteritems():
            oldtime = t.split("-")
            year = int(oldtime[0])
            month = int(oldtime[1])
            day = int(oldtime[2])
            try:
                hour = int(hourseries[i].split(":")[0])
                new_series[i] = datetime(year,  # Year
                                         month,  # Month
                                         day,  # Day
                                         hour,  # Hour
                                         0  # Minute
                )
            except:
                print("Hour series:", hourseries[i])
                print("i:",i)
                exit(-1)
        self.frames[frame_name]['timestamp'] = new_series

    def data_at_time(self, frame_name, t, return_nearest=False,
                     upper_bound=None, lower_bound=0):
        """Returns a data frame which contains only one row which has the
            specified year, month, day, and hour as provided.

        Use DataCleaner.sample_at_time instead unless you know what you're
        doing.

        Args:
            frame_name (str): Name of the frame to look at.
            t (datetime): Datetime to calculate for.
            return_nearest (bool, optional): Return the nearest row to the
                provided date, if none could be found.

        Returns:
            Returns the full row from the dataframe which matches this date.
            If multiple rows match this time, it will select the row
            arbitrarily. If no rows match, it will return None, unless
            return_nearest is True.
        """
        df = self.frames[frame_name]
        # Get the date time object to compare with.

        # set up our binary search.
        if upper_bound is None:
            max_bound = len(df)-1
        else:
            max_bound = upper_bound
        min_bound = lower_bound
        current_index = (max_bound + min_bound)//2
        while True:
            comparison_time = df.loc[current_index, 'timestamp']
            if t == comparison_time:
                return df.iloc[current_index, :], current_index
            elif max_bound == min_bound:
                if return_nearest:
                    # The max or min bound is the closest item in the dataset.
                    return df.iloc[min_bound, :], min_bound
                else:
                    # Return None, indicating that nothing of value was found.
                    return None, None
            else:
                if t < comparison_time:
                    # Our time is earlier!
                    max_bound = current_index
                    current_index = (max_bound + min_bound)//2
                else:
                    # Our time is later!
                    min_bound = current_index + 1
                    current_index = (min_bound + max_bound)//2

    def sample_at_time(self, frame_name, t,
                       return_nearest=False,
                       lower_bound=0,
                       upper_bound=None):
        """Returns the sample_measurement from the specified time. Under the
            hood, this simply calls [[data_at_time]], and extracts the
            sample_measurement value.

        Args:
            frame_name (str): Name of the frame to look at.
            t (datetime): Datetime to calculate for.
            return_nearest (bool, optional): Return the nearest row to the
                provided date, if none could be found.

        Returns:
            Returns the sample_measurement value which matches the given date.
            If multiple rows match this time, it will select the row
            arbitrarily. If no rows match, it will return None, unless
            return_nearest is True.
        """
        out = self.data_at_time(
            frame_name,
            t,
            return_nearest,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

    def sample_at_day(self, frame_name, d,
                       return_nearest=False):
        """Returns the sample_measurement from the specified day. Under the
            hood, this simply calls [[data_at_time]], and extracts the
            sampled_measurement value.

        Args:
            frame_name (str): Name of the frame to look at.
            d (datetime): Datetime to calculate for. Since we are looking for a daily value,
                the time should be 00:00:00 (all daily summaries have this as the time value)
            return_nearest (bool, optional): Return the nearest row to the
                provided date, if none could be found.

        Returns:
            Returns the sample_measurement value which matches the given date.
            If multiple rows match this time, it will select the row
            arbitrarily. If no rows match, it will return None, unless
            return_nearest is True.
        """
        return (self.data_at_time(
            frame_name,
            d,
            return_nearest
        )['first_max_value'], 
        self.data_at_time(
            frame_name,
            d,
            return_nearest
        )['arithmetic_mean']
        )

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

if __name__ == "__main__":
    cleaner = DataCleaner('query_storage')
    filenames = ['../query_storage/query_SD_no2_daily']
    for fp in filenames:
        df = pd.read_csv(fp)
        cleaner.consume_frame(df, "daily")
        # print(cleaner.frames)
