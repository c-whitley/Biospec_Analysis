import numpy as np
import pandas as pd


def convert_to_secondary(input_data, secondary_structures):

    # Check if the inputted type is a dataframe
    # from here onwards only use numpy arrays
    if isinstance(input_data, pd.DataFrame):

        input_data = input_data.values

    input_data

    return output