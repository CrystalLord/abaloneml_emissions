import matplotlib.pyplot as plt


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
    plt.show()
