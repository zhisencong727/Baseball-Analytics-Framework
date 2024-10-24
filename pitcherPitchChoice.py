import pandas as pd

def getPitchSelection(df,hand):
    
    #df = pd.read_csv(pitchLogFile)
    df_selected = df[(df['stand'] == hand)]
    pitchTypeCount = df_selected['pitch_type'].value_counts()
    dict = pitchTypeCount.to_dict()
    
    total = sum(dict.values())
    normalizedPitchSelection = {key:round(value/total,3) for key,value in dict.items()}


    return normalizedPitchSelection

#getPitchSelection("seth_lugo_pitch_log.csv","L")

        