import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_data():
    times_df_old = pd.read_csv('Time.csv')
    times_df = times_df_old[times_df_old['date']>'2020-02-19']
    times_df = times_df.reset_index(drop=True) # reset indices

    
    meanual_assessment = pd.read_csv('manual_assessment.csv')


    dates = times_df['date']
    clicks_out = times_df['clicks_out']



    

    crucial_dates_indices_good = meanual_assessment[meanual_assessment['guess']>0]['date']
    crucial_dates_indices_bad = meanual_assessment[meanual_assessment['guess']<0]['date']

    for crucial_date in crucial_dates_indices_good:
        x = times_df.index[times_df['date'] == crucial_date].values
        print(x)
        #good += times_df.index[times_df['date'] == crucial_date].tolist() 
    
        plt.axvline(x=crucial_date, color='green')
        

    #print(crucial_dates_indices_good)
    

    
    plt.plot(clicks_out)
    plt.show()

def read_news_file():
    
    def is_valid_date(line):

        months = ["januari", "februari", "mars", "april", "maj", "juni", "juli", "augusti", "september", "oktober", "november", "december"]

        valid_length = (len(line) == 2 or len(line)==3)
        #print("valid_length", valid_length)
        has_day = line[0].isdigit()
        #print("has_day", has_day)

        valid_day = has_day and int(line[0])<=31 and int(line[0])>0 
        #print("valid_day", valid_day)
        
        valid_month = valid_length and (line[1].strip() in months)
        #print("valid_month", valid_month)

        return valid_length and valid_day and valid_month
                
    lines = []
    valid_dates = []

    with open('summary_omni_covid.txt',encoding='utf-8') as f:
        lines = f.readlines()
    
    # vi vet att det är formatterat så att ett nytt datum är en egen line, och börjar med dagssiffra följd av månad

    #minst 20 är datum
    #print(len(lines[0]))

    for line in lines:
        #print(line)
        if is_valid_date(line.split(" ")):
            # we found a month in the text
            valid_dates.append(line)
        #break

    print(len(valid_dates))
    print(valid_dates)

def main():

    


    """
    TODO:
    file:///C:/Users/Carin/Downloads/2022-02-02T07_55_24.pdf.pdf
    1. manuellt bedöm en sentiment på varje av de 20 nyheterna [1,0] # correlate with Pierres "event" guesses!
    2. plotta det med covid-datan
    3. manuellt bedöm [0,10]
    4. plotta med datan igen
    5. kolla om det är rimligt att det finns ett offset
    6. gör en dictionary som triggar på vissa ordföljder så som "restriktioner lättar"
    7. försök kalibrera negativt och positivt så gott det går 
    8. very overfit to training data! 
    9. kika på hur det går med ny data
    10. kolla om man kan utöka till "flyg öppnar", "terror", "stanna hemma", etc. i tidningen 
    """

if __name__ == "__main__":
    plot_data()
