from batter import batter
from pitcher import pitcher
from enum import Enum,auto
import random


class Result(Enum):
    WALK = auto()
    STRIKE_OUT = auto()
    SINGLE = auto()
    DOUBLE = auto()
    TRIPLE = auto()
    HOME_RUN = auto()
    FORCE_OUT = auto()
    FIELD_OUT = auto()

def normalizeArr(arr):
    sum1 = sum(arr)
    if sum1 == 0.0:
        return [0.0]*len(arr)
    arr = [round(x/sum1,3) for x in arr]
    return arr

def avgDict(dict1,dict2):
    ret = {}
    for eachPitch in dict1.keys():
        ret[eachPitch] = (dict1[eachPitch]*1.0+dict2[eachPitch]*1.0)/2
    return ret

def avgArr(arr1,arr2):
    ret = []
    for i in range(len(arr1)):
        ret.append((arr1[i]*1.0+arr2[i]*1.0)/2)
    return ret
    
# returns the key of the percentage dictionary
def selectFromPercentageDict(pitcherDict,batterDict):
    percentageDict = avgDict(pitcherDict,batterDict)
    prob = random.uniform(0,0.98)
    
    currSum = 0
    for eachPitchType in percentageDict.keys():
        currSum += percentageDict[eachPitchType]
        if prob < currSum:
            return eachPitchType
    #print(currSum)
    #print("prob: ",str(prob))
    return None

# returns the indexChosen of the percentage array
def selectFromPercentageArr(pitcherArr,batterArr):
    percentageArr = avgArr(pitcherArr,batterArr)
    percentageArr = normalizeArr(percentageArr)
    prob = random.uniform(0,0.989)
    currSum = 0
    indexChosen = 0
    while indexChosen < len(percentageArr):
        currSum += percentageArr[indexChosen]
        if prob <= currSum:
            return indexChosen
        else:
            indexChosen += 1
    #print("prob is:",prob)
    #print("currSum is:",currSum)
    #print(pitcherArr)
    #print(batterArr)
    #print(percentageArr)
    #print("Triggered")
    randomIndex = int(random.uniform(-2,2))     
    if randomIndex == 0:
        prob = random.uniform(0,1)
        if  prob < 0.5:
            return 0
        return 1
    return randomIndex

def atBatSim(pitcher,batter):
    
    ball = 0
    strike = 0

    while strike < 3:
        #print("batter is:",batter.lastname)
        # ball 4 walk
        if ball == 4:
            return Result.WALK
        # choose this pitch type
        currPitch = selectFromPercentageDict(pitcher.pitchSelection,pitcher.pitchSelection)
        #print("currPitch is: ",currPitch)
        # never seen this pitch before
        if currPitch not in batter.vsPitchTypeSplits or currPitch not in batter.inPlaySplits or currPitch == None:
            #print("TRIGGERED")
            #print(currPitch)
            #print(pitcher.lastname)
            prob = random.uniform(0,1)
            if prob < 0.2:
                return Result.WALK
            if prob < 0.4:
                return Result.SINGLE
            if prob < 0.6:
                return Result.STRIKE_OUT
            return Result.FIELD_OUT
        # batter response
        batterResponse = selectFromPercentageArr(pitcher.vsPitchTypeSplits[currPitch],batter.vsPitchTypeSplits[currPitch])
        #print("batter response is: ",str(batterResponse))
        
        if batterResponse == -1:
            prob = random.uniform(0,1)
            if prob < 0.1:
                return Result.WALK
            if prob < 0.275:
                return Result.SINGLE
            if prob < 0.325:
                return Result.DOUBLE
            if prob < 0.35:
                return Result.HOME_RUN
            if prob < 0.65:
                return Result.STRIKE_OUT
            return Result.FIELD_OUT
        
        if batterResponse == 0:
            prob = random.uniform(0,1)
            if ball < 3:
                ball += 1
            else:
                if prob < 0.8:
                    return Result.WALK
                else:
                    batter_hit_into_play_response = selectFromPercentageArr(pitcher.inPlaySplits[currPitch],batter.inPlaySplits[currPitch])
                    if batter_hit_into_play_response == -1 :
                        prob = random.uniform(0,1)
                        if prob < 0.15:
                            return Result.SINGLE
                        if prob < 0.225:
                            return Result.DOUBLE
                        if prob < 0.25:
                            return Result.HOME_RUN
                        if prob < 0.6:
                            return Result.FORCE_OUT
                        return Result.FIELD_OUT
                    hit_into_play_dict = {}
                    hit_into_play_dict[0] = Result.SINGLE
                    hit_into_play_dict[1] = Result.DOUBLE
                    hit_into_play_dict[2] = Result.TRIPLE
                    hit_into_play_dict[3] = Result.HOME_RUN
                    hit_into_play_dict[4] = Result.FORCE_OUT
                    hit_into_play_dict[5] = Result.FIELD_OUT
                    #print(batter_hit_into_play_response)
                    return hit_into_play_dict[batter_hit_into_play_response]
                
            #print("BALL: ", str(ball))
       
        elif batterResponse == 1:
            if strike == 2:
                twoStrikeFoulPercentage = (pitcher.twoStrikeFoulPercentage + batter.twoStrikeFoulPercentage)/2
                prob = random.uniform(0,1)
                if prob < twoStrikeFoulPercentage:
                    #print("fouled out 2 strikes")
                    continue
                else:
                    #print("STRIKEOUT")
                    return Result.STRIKE_OUT
            else:
                strike += 1
            #print("STRIKE: ", str(strike))
        
        else:
            batter_hit_into_play_response = selectFromPercentageArr(pitcher.inPlaySplits[currPitch],batter.inPlaySplits[currPitch])

            if batter_hit_into_play_response == -1:
                prob = random.uniform(0,1)
                if prob < 0.15:
                    return Result.SINGLE
                if prob < 0.225:
                    return Result.DOUBLE
                if prob < 0.25:
                    return Result.HOME_RUN
                if prob < 0.5:
                    return Result.FORCE_OUT
                return Result.FIELD_OUT

            hit_into_play_dict = {}
            hit_into_play_dict[0] = Result.SINGLE
            hit_into_play_dict[1] = Result.DOUBLE
            hit_into_play_dict[2] = Result.TRIPLE
            hit_into_play_dict[3] = Result.HOME_RUN
            hit_into_play_dict[4] = Result.FORCE_OUT
            hit_into_play_dict[5] = Result.FIELD_OUT
            return hit_into_play_dict[batter_hit_into_play_response]
    #print("atBat over for: ",batter.lastname)
    return Result.STRIKE_OUT


def main():
    #batter1 = batter("amed","rosario","L")
    #pitcher1 = pitcher("néstor","cortés","R")
    #print(pitcher1.pitchSelection)
    #print(batter1.vsPitchTypeSplits)
    #print(pitcher1.vsPitchTypeSplitsè)

    """
    resultDict = {}
    for i in range(10000):
        temp = atBatSim(pitcher1,batter1)
        if temp in resultDict.keys():
            resultDict[temp] += 1
        else:
            resultDict[temp] = 1
    print(resultDict)
    """

        

#main()