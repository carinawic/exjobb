# dicky-fuller test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


air_passengers = pd.read_csv("datasets/AirPassengers.csv", index_col=0, parse_dates=True)
air_passengers.head()