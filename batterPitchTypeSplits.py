import pandas as pd

def getVersusPitchTypeData(df,hand, bORp):

    versusPitchTypeDict = {}
    inPlaySplitsDict = {}

    #totalPitches = 0
    pitchTypes = ['FF','CU','CH','FC','SL','SI','ST','FS','KC','SV','KN']

    #df = pd.read_csv(pitchLogFile)
    
    two_strikes_foul_percentage = 0.0

    for eachPitchType in pitchTypes:

        #print("For Pitch Type: " + eachPitchType)

        if bORp == "b":
            df_selected = df[(df['p_throws'] == hand) & (df['pitch_type'] == eachPitchType)]
        else:
            df_selected = df[(df['stand'] == hand) & (df['pitch_type'] == eachPitchType)]
        #totalPitches += len(df_selected)
        if len(df_selected) == 0:
            #print("No Corresponding Entry")
            continue
        df_fouls = df_selected[(df_selected['description'] == 'foul')]
        df_foul_strikes = df_fouls[(df_fouls['strikes'] != 2)]
        foul_strikes = len(df_foul_strikes)

        df_two_strikes = df_selected[(df_selected['strikes'] == 2)]
        df_two_strikes_foul = df_two_strikes[(df_two_strikes['description'] == 'foul')]
        two_strikes_foul_percentage = 0.0
        if len(df_two_strikes) == 0:
            two_strikes_foul_percentage = 0.0
        else:
            two_strikes_foul_percentage = round(len(df_two_strikes_foul)/len(df_two_strikes),3)
        #print("two_strike_foul_percentage: " + str(two_strikes_foul_percentage) + "%")

        total = len(df_selected)
        description_count = df_selected['description'].value_counts()
        ball_percentage = 0.0
        strike_percentage = 0.0
        hit_into_play_percentage = 0.0
        single_percentage = 0.0
        double_percentage = 0.0
        triple_percentage = 0.0
        home_run_percentage = 0.0
        force_out_percentage = 0.0
        field_out_percentage = 0.0
        gidp_percentage = 0.0

        strike_counts = df_selected[df_selected['description'].str.contains('strike', case=False)]['description'].value_counts()
        total_strikes = strike_counts.sum()
        events_count = df_selected['events'].value_counts()
        #print(description_count)
        #print(events_count)
    

        if 'ball' in description_count:
            ball_percentage = round((description_count['ball']/total),3)
        if total_strikes > 0:
            strike_percentage = round((total_strikes+foul_strikes)/total,3)
        if 'hit_into_play' in description_count:
            if 'single' not in events_count:
                single_percentage = 0.0
            else:
                single_percentage = round((events_count['single']/total),3)
            if 'double' not in events_count:
                double_percentage = 0.0
            else:
                double_percentage = round((events_count['double']/total),3)
            if 'triple' not in events_count:
                triple_percentage = 0.0
            else:
                triple_percentage = round((events_count['triple']/total),3)
            if 'home_run' not in events_count:
                home_run_percentage = 0.0
            else:
                home_run_percentage = round((events_count['home_run']/total),3)
            if 'force_out' not in events_count:
                force_out_percentage = 0.0
            else:
                force_out_percentage = round((events_count['force_out']/total),3)
            if 'field_out' not in events_count:
                field_out_percentage = 0.0
            else:
                field_out_percentage = round((events_count['field_out']/total),3)
            hit_into_play_percentage = round((description_count['hit_into_play']/total),3)

        #print(f"Ball Percentage: {ball_percentage:.2f}%")
        #print(f"Strike Percentage: {strike_percentage:2f}%")
        #print(f"Hit_Into_Play_Percentage:{hit_into_play_percentage:2f}%")

        splitPercentage = [ball_percentage,strike_percentage,hit_into_play_percentage]
        splitSum = sum(splitPercentage)
        if splitSum == 0.0:
             versusPitchTypeDict[eachPitchType] = [0.0 for y in splitPercentage]
        else:
            versusPitchTypeDict[eachPitchType] = [round(y/splitSum,3) for y in splitPercentage]

        inPlay = [single_percentage,double_percentage,triple_percentage,home_run_percentage,force_out_percentage,field_out_percentage]
        inPlaySum = sum(inPlay)
        #print(inPlaySum)
        if inPlaySum == 0.0:
            inPlaySplitsDict[eachPitchType] = [0.0,0.0,0.0,0.0,0.0,0.0]
        else:
            inPlaySplitsDict[eachPitchType] = [round(x/inPlaySum,3) for x in inPlay]

        
    
    return versusPitchTypeDict,inPlaySplitsDict,two_strikes_foul_percentage


