import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def prepareData():
    
    # read media_clicks_df as pd df and add sum_total col
    covid_numbers = pd.read_csv('owid-covid-data-sweden.csv')
    
    small = covid_numbers[['date','new_cases','new_deaths']].copy()

    small.to_csv('owid-covid-data-sweden-small.csv', index=False)

def prepareData2():
    
    # read media_clicks_df as pd df and add sum_total col
    covid_numbers = pd.read_csv('stay-at-home-covid-sweden-binary.csv')
    #new_df = covid_numbers[covid_numbers['Entity'] =='Sweden']
    last_n_column = covid_numbers.iloc[: , -2:]
    last_n_column.to_csv('stay-at-home-covid-sweden-binary.csv', index=False)



def correlate_data():
    
    times_df_old = pd.read_csv('Time.csv')
    times_df = times_df_old[times_df_old['date']>'2020-02-19']

    dates = times_df['date']
    clicks_out = times_df['clicks_out']
    
    covid_numbers_df = pd.read_csv('owid-covid-data-sweden-small.csv')
    new_cases_and_deaths = covid_numbers_df[covid_numbers_df['date']>'2020-02-19']
    new_deaths = new_cases_and_deaths['new_deaths']
    new_cases = new_cases_and_deaths['new_cases']

    new_deaths = new_deaths.fillna(0)
    new_deaths = new_deaths.clip(lower=0) # deaths cannot be a negative amount of people
    
    dates.reset_index(drop=True, inplace=True)
    clicks_out.reset_index(drop=True, inplace=True)
    new_cases.reset_index(drop=True, inplace=True)
    new_deaths.reset_index(drop=True, inplace=True)

    clicls_and_stay_req = pd.concat([dates, clicks_out, new_cases, new_deaths], axis=1)

    plt.title('Correlation between covid and clickouts')
    plt.xlabel('days')
    plt.ylabel('cases')

    plt.plot(clicls_and_stay_req['clicks_out'],'-', label="clicks_out")
    plt.plot(clicls_and_stay_req['new_cases'],'-', label="new_cases")
    plt.plot(clicls_and_stay_req['new_deaths']*50,'-', label="new_deaths*50")
    
    plt.legend()
    #plt.show()
    
    """
    # medians
    med1 = np.median(np.array(clicls_and_stay_req['new_deaths']))
    print(med1)
    
    med2 = np.median(np.array(clicls_and_stay_req['new_cases']))
    print(med2)
    """
    deadly_clickouts = clicls_and_stay_req[clicls_and_stay_req['new_deaths']>1]['clicks_out']
    print("len deadly_clickouts")
    print(len(deadly_clickouts))
    print("sum deadly_clickouts")
    print(deadly_clickouts.sum())

    undeadly_clickouts = clicls_and_stay_req[clicls_and_stay_req['new_deaths']<1]['clicks_out']
    print("len undeadly_clickouts")
    print(len(undeadly_clickouts))
    print("sum undeadly_clickouts")
    print(undeadly_clickouts.sum())


    """
    infected_clickouts = clicls_and_stay_req[clicls_and_stay_req['new_cases']>366]['clicks_out']
    print("len infected_clickouts")
    print(len(infected_clickouts))
    print("sum infected_clickouts")
    print(infected_clickouts.sum())

    unfected_clickouts = clicls_and_stay_req[clicls_and_stay_req['new_cases']<366]['clicks_out']
    print("len unfected_clickouts")
    print(len(unfected_clickouts))
    print("sum unfected_clickouts")
    print(unfected_clickouts.sum())"""

if __name__ == "__main__":
    correlate_data()