import pandas as pd

def make_time_range(start_time, end_time, freq):
    time_range = pd.date_range(start_time, end_time, freq=freq)
    return pd.DataFrame(time_range, columns=["ds"])
