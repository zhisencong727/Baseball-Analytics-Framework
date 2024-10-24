import pandas as pd
from pybaseball import playerid_lookup,playerid_reverse_lookup
from pybaseball import statcast_batter
from batterPitchTypeSplits import getVersusPitchTypeData
from getSplitCoeffcients import getSplitCoeffcient
import os

def normalizeArr(arr):
    sum1 = sum(arr)
    if sum1 == 0.0:
        return [0.0]*len(arr)
    arr = [round(x/sum1,3) for x in arr]
    return arr

class batter:
    def __init__(self,firstname,lastname, hand, input_mlbam,day_night,home_away):
        self.firstname = firstname
        self.lastname = lastname
        self.hand = hand
        df = playerid_lookup(self.lastname,self.firstname,fuzzy=True)
        print("Batter is: " + self.firstname + self.lastname)
        self.keyDict = df.to_dict()
    
    #def getPitchLog(self):
        for index, year in self.keyDict['mlb_played_last'].items():
            if year == 2024:
        # Get the mlbam ID for this player
                mlbam_id = self.keyDict['key_mlbam'][index]
                self.mlbam_id = mlbam_id
                self.bbref_id = self.keyDict['key_bbref'][index]
                break
        #print(mlbam_id)
        #mlbam_id = next(iter(self.keyDict['key_mlbam'].values()))
        if input_mlbam != 0:
            mlbam_id = input_mlbam
            self.mlbam_id = mlbam_id
            dict1 = playerid_reverse_lookup([self.mlbam_id],key_type='mlbam').to_dict()
            self.bbref_id = next(iter(dict1['key_bbref'].values()))

        print(mlbam_id)
        df = statcast_batter('2024-3-28','2024-8-22',player_id=mlbam_id)
        """
        pitchLogFileName = self.firstname+"_"+self.lastname+"_"+"pitch_log.csv"
        pitchLogFileName = os.path.join("pitchLogFiles",pitchLogFileName)
        df.to_csv(pitchLogFileName,index=True)
        """
        self.vsPitchTypeSplits,self.inPlaySplits,self.twoStrikeFoulPercentage = getVersusPitchTypeData(df,self.hand,'b')
        
        #for each in self.vsPitchTypeSplits.keys():
            #print(each)
            #print(self.vsPitchTypeSplits[each])
        #print(self.inPlaySplits)

        
        splitCoeffcient = getSplitCoeffcient(self.firstname,self.lastname,self.bbref_id,day_night,home_away,False)

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
            #print(self.inPlaySplits[eachPitch])

        self.twoStrikeFoulPercentage *= splitCoeffcient

        self.splitCoefficient = splitCoeffcient

        self.single = 0
        self.double = 0
        self.triple = 0
        self.home_run = 0
        self.walk = 0
        self.strikeout = 0
        self.in_play_outs = 0
        self.rbi = 0

        self.splitCoefficient = splitCoeffcient
        


def main():
    # Hand here is the opposing pitcher's throwing hand
    batter1 = batter("elly","de la cruz","R",0,"day","away")
    #batter1.getPitchLog()
    #print(batter1.vsPitchTypeSplits)
    #print(batter1.inPlaySplits)
#main()