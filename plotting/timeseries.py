import matplotlib.pyplot as plt
import argparse
import datetime
import numpy as np
import matplotlib.dates as mdates

def timeseries(x, y, title='Time Series', ylabel='Value', plttype='line',
               time_range=None, legend=None):
    """Plot a time series in matplot lib

    Args:
        x (array-like): X positions of datapoints.
        y (array-like): Y positions of datapoints.
        title (str, optional): Title of the plot. Default is 'Time Series'.
        ylabel (str, optional): ylabel of the plot. Default is 'Value'.
        plttype (str, optional): Can be 'line' or 'stem'. Default is 'line'.
        time_range (tuple, optional): Time range to look at. Default is the
            entire range.
        legend (tuple, optional): Tuple of strings to set the legend.
    """

    # Select only a specific time range to look at.
    if time_range is not None:
        num_x = len(x)
        min_index = 0
        max_index = 0
        for i in range(num_x):
            if x[i] >= time_range[0]:
                min_index = i
                break
        for i in reversed(range(num_x)):
            if x[i] < time_range[1]:
                max_index = i
                break
        selx = x[min_index:max_index+1]
        sely = y[min_index:max_index+1]
    else:
        # Selected X & Y
        selx = x
        sely = y

    if plttype == "line":
        plt.plot(selx, sely)
    else:
        plt.stem(selx, sely)

    # Set up legend if it exists.
    if legend is not None:
        plt.legend(legend)

    plt.ylabel(ylabel)
    plt.xlabel('Time')
    plt.title(title)
    plt.grid(alpha=0.1)

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    # plt.plot(x,y)
    # plt.gcf().autofmt_xdate()

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np_array", type=str, help='path to compiled numpy file')
    # Add argument specifying what value you want to plot
    args = vars(parser.parse_args())

    arr = args["np_array"]
    df_array = np.load(arr)
# NOTE: columns will be named based on the query, so there will be things like:
# VOC_parameter_name, ozone_parameter_name, etc.
    date_ind = 0 #np.where(df_array[0] =='date_local')
    time_ind = 1 #np.where(df_array[0] =='time_local')
    param_ind = 2 #np.where(df_array[0] =='parameter_name')
    samp_ind = 3 #np.where(df_array[0] =='sample_measurement')
    params = np.unique(df_array[:,param_ind]) # get an array of unique parameters
    for param in params:
        timeList = []
        valList = []
        for samp in df_array:
            if samp[param_ind] == param:
                if samp[date_ind].month > 5 and samp[date_ind].month < 9:
                    time = datetime.datetime.combine(samp[date_ind], datetime.datetime.strptime(samp[time_ind], '%H:%M').time())
                    timeList.append(time)
                    valList.append(samp[samp_ind])   
        print(timeList)
        print(valList)
        timeseries(timeList, valList, title=param + " levels vs. time")


if __name__ == "__main__":
    main()

