import pandas as pd
import numpy as np
import polars as pl
import plotly.graph_objects as go


def get_time_plot_frame(df_data:pl.LazyFrame, time_delta:float, behaviors:list[str],
                        sensor_features:list[str], time_step_size:float=0.01) -> pd.DataFrame:
    """
    Generates a DataFrame with consecutive time samples of either each behavior or each sensor
    (one of both inputs must be a list of length 1).
    Args:
        df_data: Cleaned DataFrame with time information.
        time_delta: Length of required time interval.
        behaviors: Behaviors for which a time interval shall be found.
        sensor_features: Sensor for which a time interval shall be found.
        time_step_size: Time delta between each time measurement.
    """
    # calculate number of time steps
    num_steps = round(time_delta/time_step_size)+1

    # initialize time plot DataFrame
    time_plot_df = pd.DataFrame({'t_sec': np.linspace(0, time_delta, num_steps)})

    # iterate over each sensor type
    if len(behaviors) == 1:
        behavior = behaviors[0]
        for sensor in sensor_features:
            # get all time steps for the given behavior and curent sensor
            tmp_df = df_data.select(['t_sec', sensor, 'Behavior']).filter(
                pl.col('Behavior')==behavior).collect().to_pandas()
        
            # get starting index of a valid consecutive time series
            idx = _get_valid_time_series(tmp_df, num_steps)

            # add column to DataFrame with valid time series for the current sensor
            time_plot_df[sensor] = tmp_df[sensor].iloc[idx:idx+num_steps].to_list()
        return time_plot_df.melt(id_vars=['t_sec'], value_vars=sensor_features)
    # iterate over each behavior
    elif len(sensor_features) == 1:
        sensor = sensor_features[0]
        for behavior in behaviors:
            # get all time data for the given sensor and current behavior
            tmp_df = df_data.select(['t_sec', sensor, 'Behavior']).filter(
                pl.col('Behavior')==behavior).collect().to_pandas()
        
            # get starting index of a consecutive valid time series
            idx = _get_valid_time_series(tmp_df, num_steps)

            # add column to DataFrame with valid time series with the current behavior
            time_plot_df[behavior] = tmp_df[sensor].iloc[idx:idx+num_steps].to_list()
        return time_plot_df.melt(id_vars=['t_sec'], value_vars=behaviors)

    raise ValueError('One of the two input lists must have length 1.')

def _get_valid_time_series(df_time:pd.DataFrame, num_steps:int):
    """
    Finds the starting index of a consecutive time series of the sepecified length.
    """
    len_test = len(df_time)

    i=0
    while i < len_test - num_steps:
        # check if all given time steps are consecutive
        idx = np.argmax([False, *(~np.isclose(np.diff(df_time['t_sec'].iloc[i:i+num_steps]), 0.01))])
        if idx == 0:
            # valid time sequence has been found, break out of loop
            break
        else:
            # the time sequence is not valid; move to the index that breaks continuity
            i += idx

    return i

def make_manual_box_plot(exploded_df:pl.LazyFrame, sensor_feature:str, title:str='', xlabel:str='', ylabel:str=''):
    """
    Returns a manually contstructed Boxplot-figure.
    """
    # save grouped DataFrame as interim result
    grouped = exploded_df.select(['Behavior', sensor_feature]).group_by('Behavior')
    # initialize figure object
    fig = go.Figure()
    # add all behavior categories to the x-axis
    fig.add_trace(go.Box(x=grouped.first().select('Behavior').sort('Behavior').collect().to_series().to_list()))
    # calculate quantiles and mean and add to figure
    fig.update_traces(
        q1=grouped.quantile(0.25).sort('Behavior').select(sensor_feature).collect().to_series().to_list(),
        median=grouped.median().sort('Behavior').select(sensor_feature).collect().to_series().to_list(),
        q3=grouped.quantile(0.75).sort('Behavior').select(sensor_feature).collect().to_series().to_list(),
        lowerfence=grouped.quantile(0.1).sort('Behavior').select(sensor_feature).collect().to_series().to_list(),
        upperfence=grouped.quantile(0.9).sort('Behavior').select(sensor_feature).collect().to_series().to_list(),
        mean=grouped.mean().sort('Behavior').select(sensor_feature).collect().to_series().to_list(),
    )
    # add axes labels and title to figure
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        title=title
    )
    # return figure
    return fig