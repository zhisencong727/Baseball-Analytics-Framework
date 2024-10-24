import pandas as pd
from pybaseball import playerid_lookup,playerid_reverse_lookup
from pybaseball import statcast_pitcher
from pitcherPitchChoice import getPitchSelection
from batterPitchTypeSplits import getVersusPitchTypeData
from getSplitCoeffcients import getSplitCoeffcient
import os

def normalizeArr(arr):
    sum1 = sum(arr)
    if sum1 == 0.0:
        return [0.0]*len(arr)
    arr = [round(x/sum1,3) for x in arr]
    return arr

class pitcher():
    def __init__(self,firstname,lastname,hand,input_mlbam,day_night,home_away):
        self.firstname = firstname
        self.lastname = lastname
        self.hand = hand
        df = playerid_lookup(self.lastname,self.firstname,fuzzy=True)
        print("Pitcher is: " + self.firstname + self.lastname)
        self.keyDict = df.to_dict()
    
        for index, year in self.keyDict['mlb_played_last'].items():
            if year == 2024:
        # Get the mlbam ID for this player
                mlbam_id = self.keyDict['key_mlbam'][index]
                self.mlbam_id = mlbam_id
                self.bbref_id = self.keyDict['key_bbref'][index]
                break
        print(mlbam_id)
    #def getPitchLog(self):
        #mlbam_id = next(iter(self.keyDict['key_mlbam'].values()))
        if input_mlbam != 0:
            mlbam_id = input_mlbam
            self.mlbam_id = mlbam_id
            dict1 = playerid_reverse_lookup([self.mlbam_id],key_type='mlbam').to_dict()
            self.bbref_id = next(iter(dict1['key_bbref'].values()))
        

        df = statcast_pitcher('2024-3-28','2024-8-22',player_id=mlbam_id)
        
        pitchLogFileName = self.firstname+"_"+self.lastname+"_"+"pitch_log.csv"
        pitchLogFileName = os.path.join("pitchLogFiles",pitchLogFileName)
        df.to_csv(pitchLogFileName,index=True)
        
    #def getPitchSelection(self):
        #pitchLogFileName = self.firstname+"_"+self.lastname+"_"+"pitch_log.csv"
        self.pitchSelection = getPitchSelection(df,self.hand)
    
    #def getPitchSplits(self):
        self.vsPitchTypeSplits,self.inPlaySplits,self.twoStrikeFoulPercentage = getVersusPitchTypeData(df,self.hand,'p')

        #print(self.vsPitchTypeSplits)
        #print(self.inPlaySplits)

        
        splitCoeffcient = getSplitCoeffcient(self.firstname,self.lastname,self.bbref_id,day_night,home_away,True)

        for eachPitch in self.vsPitchTypeSplits.keys(): 
            weightedVsPitchTypeSplits = self.vsPitchTypeSplits[eachPitch]
            weightedVsPitchTypeSplits[0] *= splitCoeffcient
            weightedVsPitchTypeSplits[1] *= (1/splitCoeffcient)
            weightedVsPitchTypeSplits[2] *= splitCoeffcient

            self.vsPitchTypeSplits[eachPitch] = normalizeArr(weightedVsPitchTypeSplits)
            #print(self.vsPitchTypeSplits[eachPitch])

        for eachPitch in self.inPlaySplits.keys():
            weightedInPlaySplits = self.inPlaySplits[eachPitch]
            weightedInPlaySplits[0] *= splitCoeffcient
            weightedInPlaySplits[1] *= splitCoeffcient
            weightedInPlaySplits[2] *= splitCoeffcient
            weightedInPlaySplits[3] *= splitCoeffcient
            weightedInPlaySplits[4] *= (1/splitCoeffcient)
            weightedInPlaySplits[5] *= (1/splitCoeffcient)

            self.inPlaySplits[eachPitch] = normalizeArr(weightedInPlaySplits)
            #print(self.vsPitchTypeSplits[eachPitch])

        self.twoStrikeFoulPercentage *= splitCoeffcient

        self.splitCoefficient = splitCoeffcient
        
        
        

        
def main():
    # hand here is also opposing player's hand, maybe change this later when
    # we have both pitcher and batter in a matchup
    pitcher1 = pitcher("nestor","cortes","L",0,"day","away")
    #pitcher1.getPitchLog()
    #pitcher1.getPitchSelection()
    #pitcher1.getPitchSplits()
    #print(pitcher1.pitchSelection)
    #print(pitcher1.vsPitchTypeSplits)
    #print(pitcher1.inPlaySplits)

main()
