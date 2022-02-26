import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def readData():
    
    stay_home_df = pd.read_csv('stay-at-home-covid-sweden-binary.csv')
    stay_home_requirements = np.array(stay_home_df['stay_home_requirements'].values) * 10000

    new_cases_df = pd.read_csv('owid-covid-data-sweden-small.csv')
    new_cases = np.array(new_cases_df['new_cases'].values)
    new_deaths = np.array(new_cases_df['new_deaths'].values) * 100

    plt.plot(stay_home_requirements,'-x', label="stay home")
    plt.plot(new_cases,'-x', label="new cases")
    plt.plot(new_deaths,'-x', label="new deaths")
    
    plt.legend()

    plt.show()

if __name__ == "__main__":
    readData()