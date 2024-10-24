from pybaseball import get_splits,playerid_lookup
import pandas as pd
import warnings
from bs4 import BeautifulSoup


# Suppress specific warning


def getSplitCoeffcient(firstname,lastname,bbref_id,day_night,home_away,pitcher):

    if pitcher:
        df,_ = get_splits(bbref_id,pitching_splits=pitcher,year=2024)
    else:
        df = get_splits(bbref_id,year=2024)

    df.to_csv("playerSplits/" + firstname + lastname + ".csv")


    # Replace 'file_path.csv' with the path to your CSV file
    df = pd.read_csv("playerSplits/" + firstname + lastname + ".csv")
    home_ops = df[(df['Split'] == "Home")]['OPS'].to_dict().values()
    if len(home_ops) == 0:
        home_ops = 2.0
    else:
        home_ops = list(home_ops)[0]

    away_ops = df[(df['Split'] == "Away")]['OPS'].to_dict().values()
    if len(away_ops) == 0:
        away_ops = 2.0
    else:
        away_ops = list(away_ops)[0]

    home_away_coefficient = 0.0
    if home_ops == 2.0 or away_ops == 2.0:
        home_away_coefficient = 1.0
    elif home_away == "home":
        home_away_coefficient = home_ops/((home_ops+away_ops)/2)
    else:
        home_away_coefficient = away_ops/((home_ops+away_ops)/2)

    day_ops = df[(df['Split'] == "Day")]['OPS'].to_dict().values()
    if len(day_ops) == 0:
        day_ops = 2.0
    else:
        day_ops = list(day_ops)[0]

    night_ops = df[(df['Split'] == "Night")]['OPS'].to_dict().values()
    if len(night_ops) == 0:
        night_ops = 2.0
    else:
        night_ops = list(night_ops)[0]
    
    day_night_coefficient = 0.0
    if day_ops == 2.0 or night_ops == 2.0:
        day_night_coefficient == 1.0
    elif day_night == "day":
        day_night_coefficient = day_ops/((day_ops+night_ops)/2)
    else:
        day_night_coefficient = night_ops/((day_ops+night_ops)/2)
    if day_night_coefficient == 0.0:
        day_night_coefficient = 1.0
    first5_ops = df[(df['Split'] == "1st Half")]['OPS'].to_dict().values()
    if len(first5_ops) == 0:
        first5_ops = 2.0
    else:
        first5_ops = list(first5_ops)[0]

    second5_ops = df[(df['Split'] == "2nd Half")]['OPS'].to_dict().values()
    if len(second5_ops) == 0:
        second5_ops = 2.0
    else:
        second5_ops = list(second5_ops)[0]

    if first5_ops == 2.0 or second5_ops == 2.0:
        first5_coefficient = 1.0
    else:
        first5_coefficient = first5_ops/((first5_ops+second5_ops)/2)
    
    seasonTotal_ops = df[(df['Split'] == "2024 Totals")]['OPS'].to_dict().values()
    seasonTotal_ops = list(seasonTotal_ops)[0]
    last14_ops = df[(df['Split'] == "Last 14 days")]['OPS'].to_dict().values()
    if len(last14_ops) == 0:
        last14_ops = 0.0
    else:
        last14_ops = list(last14_ops)[0]

    if seasonTotal_ops == 0.0 or last14_ops == 0.0:
        recency_coefficient = 1.0
    else:
        recency_coefficient = last14_ops/((last14_ops+seasonTotal_ops)/2)
    
    #print(first5_coefficient)
    #print(recency_coefficient)
    #print(day_night_coefficient)
    #print(home_away_coefficient)
    splitCoef = round(first5_coefficient*recency_coefficient*day_night_coefficient*home_away_coefficient,3)

    print("splitCoef is: ",splitCoef)
    
    return splitCoef

    
    


#print(getSplitCoeffcient("brooks","baldwin","baldwbr01","home","day",False))
#print(getSplitCoeffcient(batter1,"away","night"))

