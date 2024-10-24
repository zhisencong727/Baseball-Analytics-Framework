from batter import batter
from pitcher import pitcher
from atBatSimulation import atBatSim,Result
from teamConstructor import teamConstructor
from getSplitCoeffcients import getSplitCoeffcient
import random
import warnings

class team:
    def __init__(self,teamname,lineup,sp):
        self.teamname = teamname
        self.lineup = lineup
        self.sp = sp

def game(away,home):
    away_currBatterIndex = 0
    home_currBatterIndex = 0
    away_outs = 0
    home_outs = 0
    away_score = 0
    home_score = 0
    first_base = "Empty"
    second_base = "Empty"
    third_base = "Empty"
    
    # for each of the first 5 innings
    for inning in range(1,6):
        
        #print("inning: ", str(inning))
        # away team bat first
        while away_outs < 3:
            currAwayBatter = away.lineup[away_currBatterIndex%9]
            #print("currBatter is: ", currAwayBatter.firstname)
            result = atBatSim(home.sp[away_currBatterIndex%9],currAwayBatter)

            #print(result)
            #print("atBat over")
            away_currBatterIndex += 1
            
            # Strike_Out
            if result == Result.STRIKE_OUT:
                away_outs += 1
                currAwayBatter.strikeout += 1
                    
            # Walk
            if result == Result.WALK:
                currAwayBatter.walk += 1
                if first_base == "Occupied":
                    if second_base == "Occupied":
                        if third_base == "Occupied":
                            away_score += 1
                            currAwayBatter.rbi += 1
                        else:
                            third_base = "Occupied"
                    else:
                        second_base = "Occupied"
                else:
                    first_base = "Occupied"
            
            # Single
            if result == Result.SINGLE:
                currAwayBatter.single += 1
                # scenarios
                if third_base == "Occupied":
                    currAwayBatter.rbi += 1
                    away_score += 1
                    third_base = "Empty"
                    #print("third_base scores from single")
                if second_base == "Occupied":
                    prob = random.uniform(0,1)
                    if prob < 0.7:
                        currAwayBatter.rbi += 1
                        away_score += 1
                        second_base = "Empty"
                        #print("second_base scores from single")
                    else:
                        third_base = "Occupied"
                        second_base = "Empty"
                        #print("second_base goes to third_base from single")
                        
                if first_base == "Occupied":
                    if third_base == "Occupied":
                        second_base = "Occupied"
                    else:
                        prob = random.uniform(0,1)
                        if prob < 0.65:
                            third_base = "Occupied"
                        else:
                            second_base = "Occupied"
                first_base = "Occupied"
            
            # Double
            if result == Result.DOUBLE:
                currAwayBatter.double += 1
                if third_base == "Occupied":
                    currAwayBatter.rbi += 1
                    away_score += 1
                    third_base = "Empty"
                if second_base == "Occupied":
                    currAwayBatter.rbi += 1
                    away_score += 1
                    second_base = "Empty"
                if first_base == "Occupied":
                    prob = random.uniform(0,1)
                    if prob < 0.75:
                        third_base = "Occupied"
                        first_base = "Empty"
                    else:
                        currAwayBatter.rbi += 1
                        away_score += 1
                        first_base = "Empty"
                second_base = "Occupied"
            
            # Triple
            if result == Result.TRIPLE:
                currAwayBatter.triple += 1
                if first_base == "Occupied":
                    currAwayBatter.rbi += 1
                    away_score += 1
                    first_base = "Empty"
                if second_base == "Occupied":
                    currAwayBatter.rbi += 1
                    away_score += 1
                    second_base = "Empty"
                if third_base == "Occupied":
                    currAwayBatter.rbi += 1
                    away_score += 1
                third_base = "Occupied"
                #print("TRIPLE")
            
            # Home Run
            if result == Result.HOME_RUN:
                currAwayBatter.home_run += 1
                currAwayBatter.rbi += 1
                if first_base == "Occupied":
                    currAwayBatter.rbi += 1
                    away_score += 1
                    first_base = "Empty"
                    #print("first_base score from HR")
                if second_base == "Occupied":
                    currAwayBatter.rbi += 1
                    away_score += 1
                    second_base = "Empty"
                    #print("second_base score from HR")
                if third_base == "Occupied":
                    currAwayBatter.rbi += 1
                    away_score += 1
                    third_base = "Empty"
                    #print("third_base score from HR")
                away_score += 1
                #print("HOME RUN")
            
            
            # Field_Out
            if result == Result.FIELD_OUT:
                currAwayBatter.in_play_outs += 1
                
                prob = random.uniform(0,1)
                
                # runner on 3rd with less than 2 outs
                if third_base == "Occupied":
                    # sac fly
                    if away_outs < 2:
                        if prob > 0.7:
                            away_score += 1
                            currAwayBatter.rbi += 1
                            third_base = "Empty"
                            #print("third base scores on sac fly or fielder's choice")
                        if prob < 0.05:
                            third_base = "Empty"
                            
                # runner on 2nd and 3rd is open            
                elif second_base == "Occupied" and third_base == "Empty":
                    # ground ball advance runners from 2nd to 3rd
                    if prob > 0.75:
                        third_base = "Occupied"
                        if first_base == "Occupied":
                            first_base = "Empty"
                        else:
                            second_base = "Empty"
                
                # runner on 1st with 2nd open
                elif first_base == "Occupied" and second_base == "Empty":
                    # out at 1st, runner advances
                    if prob > 0.75:
                        second_base = "Occupied"
                        first_base = "Empty"
                    # gidp
                    elif prob < 0.05:
                        first_base = "Empty"
                    
                away_outs += 1
            
            if result == Result.FORCE_OUT:
                currAwayBatter.in_play_outs += 1
                if first_base == "Empty" and second_base == "Empty" and third_base == "Empty":
                    away_outs += 1
                else:
                    if third_base == "Occupied":
                        if second_base == "Empty":
                            third_base = "Empty"
                            if first_base == "Occupied":
                                second_base = "Occupied"
                        else:
                            if first_base == "Empty":
                                second_base = "Empty"
                        
                    elif second_base == "Occupied":
                        prob = random.uniform(0,1)
                        if first_base == "Occupied":
                            # make the out at 2nd
                            if prob < 0.7:
                                third_base = "Occupied"
                                second_base = "Empty"
                        else:
                            second_base = "Empty"
                    
                    away_outs += 1
                    first_base = "Occupied"
            
            """"    
            print("first_base: ", first_base)
            print("second_base: ", second_base)
            print("third_base: ", third_base)
            """
                
        away_outs = 0
        first_base = "Empty"
        second_base = "Empty"
        third_base = "Empty"
        
        ##### home team bats second
        
        if inning == 9 and home_score > away_score:
            return [away_score,home_score]

        #print([away_score,home_score])
        #("__________________________")
        
        
        while home_outs < 3:
            currHomeBatter = home.lineup[home_currBatterIndex%9]
            #print("currHomeBatter is: ", currHomeBatter.firstname)
            result = atBatSim(away.sp[home_currBatterIndex%9],currHomeBatter)
            #print(result)
            #print("atBat over")
            home_currBatterIndex += 1
            
            # Strike_Out
            if result == Result.STRIKE_OUT:
                currHomeBatter.strikeout += 1
                home_outs += 1
                    
            # Walk
            if result == Result.WALK:
                currHomeBatter.walk += 1
                if first_base == "Occupied":
                    if second_base == "Occupied":
                        if third_base == "Occupied":
                            home_score += 1
                            currHomeBatter.rbi += 1
                        else:
                            third_base = "Occupied"
                    else:
                        second_base = "Occupied"
                else:
                    first_base = "Occupied"
            
            # Single
            if result == Result.SINGLE:
                currHomeBatter.single += 1
                # scenarios
                if third_base == "Occupied":
                    currHomeBatter.rbi += 1
                    home_score += 1
                    third_base = "Empty"
                    #print("third_base scores")
                if second_base == "Occupied":
                    prob = random.uniform(0,1)
                    if prob < 0.7:
                        currHomeBatter.rbi += 1
                        home_score += 1
                        second_base = "Empty"
                        #print("second_base scores")
                    else:
                        third_base = "Occupied"
                        second_base = "Empty"
                        #print("second_base goes to third_base")
                if first_base == "Occupied":
                    if third_base == "Occupied":
                        second_base = "Occupied"
                    else:
                        prob = random.uniform(0,1)
                        if prob < 0.65:
                            third_base = "Occupied"
                        else:
                            second_base = "Occupied"
                first_base = "Occupied"
            
            # Double
            if result == Result.DOUBLE:
                currHomeBatter.double += 1
                if third_base == "Occupied":
                    home_score += 1
                    currHomeBatter.rbi += 1
                    third_base = "Empty"
                if second_base == "Occupied":
                    home_score += 1
                    currHomeBatter.rbi += 1
                    second_base = "Empty"
                if first_base == "Occupied":
                    prob = random.uniform(0,1)
                    if prob < 0.75:
                        third_base = "Occupied"
                        first_base = "Empty"
                    else:
                        currHomeBatter.rbi += 1
                        home_score += 1
                        first_base = "Empty"
                second_base = "Occupied"
            
            # Triple
            if result == Result.TRIPLE:
                currHomeBatter.triple += 1
                if first_base == "Occupied":
                    currHomeBatter.rbi += 1
                    home_score += 1
                    first_base = "Empty"
                if second_base == "Occupied":
                    currHomeBatter.rbi += 1
                    home_score += 1
                    second_base = "Empty"
                if third_base == "Occupied":
                    currHomeBatter.rbi += 1
                    home_score += 1
                third_base = "Occupied"
                #print("TRIPLE")
            
            # Home Run
            if result == Result.HOME_RUN:
                currHomeBatter.home_run += 1
                if first_base == "Occupied":
                    currHomeBatter.rbi += 1
                    home_score += 1
                    first_base = "Empty"
                if second_base == "Occupied":
                    home_score += 1
                    currHomeBatter.rbi += 1
                    second_base = "Empty"
                if third_base == "Occupied":
                    home_score += 1
                    currHomeBatter.rbi += 1
                    third_base = "Empty"
                currHomeBatter.rbi += 1
                home_score += 1
                #print("HOME RUN")
            
            
            
            # Field_Out
            if result == Result.FIELD_OUT:
                currHomeBatter.in_play_outs += 1
                prob = random.uniform(0,1)
                # runner on 3rd with less than 2 outs
                if third_base == "Occupied":
                    # sac fly and bring 3rd to home
                    if home_outs < 2:
                        if prob > 0.7:
                            home_score += 1
                            third_base = "Empty"
                            currHomeBatter.rbi += 1
                            #print("scored from fielder's choice")
                        if prob < 0.05:
                            third_base = "Empty"
                            
                # runner on 2nd and 3rd is open            
                elif second_base == "Occupied" and third_base == "Empty":
                    # ground ball advance runners from 2nd to 3rd
                    if prob > 0.8:
                        third_base = "Occupied"
                        if first_base == "Occupied":
                            first_base = "Empty"
                        else:
                            second_base = "Empty"
                
                # runner on 1st with 2nd open
                elif first_base == "Occupied" and second_base == "Empty":
                    # out at 1st, runner advances
                    if prob > 0.75:
                        second_base = "Occupied"
                        first_base = "Empty"
                    # gidp
                    elif prob < 0.05:
                        first_base = "Empty"
                    
                
                home_outs += 1
            
            if result == Result.FORCE_OUT:
                currHomeBatter.in_play_outs += 1
                if first_base == "Empty" and second_base == "Empty" and third_base == "Empty":
                    home_outs += 1
                else:
                    if third_base == "Occupied":
                        if second_base == "Empty":
                            third_base = "Empty"
                            if first_base == "Occupied":
                                second_base = "Occupied"
                        else:
                            if first_base == "Empty":
                                second_base = "Empty"
                        
                    elif second_base == "Occupied":
                        prob = random.uniform(0,1)
                        if first_base == "Occupied":
                            # make the out at 2nd
                            if prob < 0.7:
                                third_base = "Occupied"
                                second_base = "Empty"
                        else:
                            second_base = "Empty"
                    
                    home_outs += 1
                    
                    first_base = "Occupied"
            
            """"
            print("first_base: ", first_base)
            print("second_base: ", second_base)
            print("third_base: ", third_base)
            """
                
        home_outs = 0
        first_base = "Empty"
        second_base = "Empty"
        third_base = "Empty"

        #print([away_score,home_score])
        
        #print("___________________________")
        
    return [away_score,home_score]
    
def getBatterStats(currBatter):

    #print(currBatter.lastname)
    #print(currBatter.walk)
    #print(currBatter.single)
    totalAB = currBatter.single + currBatter.double + currBatter.triple + currBatter.home_run + currBatter.walk + currBatter.strikeout + currBatter.in_play_outs
    #print(totalAB)
    avg = (currBatter.single + currBatter.double + currBatter.triple + currBatter.home_run) / (totalAB-currBatter.walk)
    obp = (currBatter.single + currBatter.double + currBatter.triple + currBatter.home_run + currBatter.walk) / totalAB
    slg = (currBatter.single + currBatter.double*2 + currBatter.triple*3 + currBatter.home_run*4) / (totalAB-currBatter.walk)
    walk_percentage = currBatter.walk / totalAB
    strikeout_percentage = currBatter.strikeout / totalAB
    rbi = currBatter.rbi / 100000
    dict = {}
    dict["name"] = currBatter.firstname + currBatter.lastname
    dict["avg"] = round(avg,3)
    dict["obp"] = round(obp,3)
    dict["slg"] = round(slg,3)
    dict["bb%"] = round(walk_percentage,3)
    dict["k%"] = round(strikeout_percentage,3)
    dict["rbi"] = round(rbi,3)
    dict["hr"] = round(currBatter.home_run/100000,6)
    dict["splitCoef"] = round(currBatter.splitCoefficient,3)
    
    return dict
    
def getStrikeouts(currLineup):
    totalSO = 0
    for eachPlayer in currLineup:
        totalSO += eachPlayer.strikeout
    return round(totalSO/100000,3)

def main():
    
    away,home = teamConstructor("night","teamRoster/8_23/giants.txt","teamRoster/8_23/mariners.txt","R","R","SFG","SEA")
    awayWins = 0
    homeWins = 0
    ties = 0
    awayTotal = 0
    homeTotal = 0
    away_plusOneAndHalf = 0
    home_plusOneAndHalf = 0
    away_miunsOneAndHalf = 0
    home_minusOneAndHalf = 0
    print("Starting Simulation")

    for i in range(100000):
        if i % 10000 == 0:
            print(i)
        result = game(away,home)
        awayTotal += result[0]
        homeTotal += result[1]
        
        if result[0] > result[1]:
            awayWins += 1
            away_plusOneAndHalf +=1
            if result[1]+1.5 > result[0]:
                home_plusOneAndHalf += 1
            else:
                away_miunsOneAndHalf += 1

        elif result[0] < result[1]:
            homeWins += 1
            home_plusOneAndHalf += 1
            if result[0]+1.5 > result[1]:
                away_plusOneAndHalf += 1
            else:
                home_minusOneAndHalf += 1
        
        else:
            ties += 1
            away_plusOneAndHalf += 1
            home_plusOneAndHalf += 1


    print(away.teamname + " Wins: ", str(awayWins))
    print(away.teamname + "-1.5: ",str(away_miunsOneAndHalf))
    print(away.teamname + "+1.5: ",str(away_plusOneAndHalf))
    print(away.teamname + "+0.5: ",str(awayWins + ties))
    print(home.teamname + " Wins: ", str(homeWins))
    print(home.teamname + "-1.5: ",str(home_minusOneAndHalf))
    print(home.teamname + "+1.5: ",str(home_plusOneAndHalf))
    print(home.teamname + "+0.5: ",str(homeWins + ties))
    print("ties: ",str(ties))
    print(away.teamname + " Avg: ", str(round(awayTotal/100000,3)))
    print(home.teamname + " Avg: ", str(round(homeTotal/100000,3))) 
    
    for eachAwayBatter in away.lineup:
        print(getBatterStats(eachAwayBatter))
    for eachHomeBatter in home.lineup:
        print(getBatterStats(eachHomeBatter))
    
    print(away.sp[0].lastname,end=" strikeouts: ")
    print(getStrikeouts(home.lineup))

    print(home.sp[0].lastname,end=" strikeouts: ")
    print(getStrikeouts(away.lineup))


main()
        
