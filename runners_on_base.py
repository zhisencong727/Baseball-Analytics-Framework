# stats from teamranking.com Houston Astros
# 34.31 at bats + 2.78 walks + 0.42 hbp + roughly 0.5 sacrafices comes out to around 38 PAs per game
# walk rate 2.78 walk per game = 7.41%
# homerun rate 1.17 hr per home run = 3.1%
# Single 6.07 singles per game = 16.0%
# Double 1.63 doubles per game = 4.3%
# Triple 0.10 triples per game = 0.26%
# Double play 0.83 per game = 2.1%
# Batting Avg with RISP 26.70%
# Advances runner through sac fly or grounding out = 20%
from enum import Enum,auto
class Result(Enum):
    WALK = auto()
    SINGLE = auto()
    DOUBLE = auto()
    TRIPLE = auto()
    HOME_RUN = auto()
    DOUBLE_PLAY = auto()
    OUT = auto()
    STRIKE_OUT = auto()



import random
def inning(current_batter,num_innings):
    runs = 0
    batting_avg_coef = [1.20,1.12,1.08,1.04,1.00,0.96,0.92,0.88,0.80] # [1.20,1.12,1.08,1.04,1.00,0.96,0.92,0.88,0.80]
    walk_coef = [1.04,1.08,1.12,1.24,1.04,0.96,0.92,0.84,0.76] # [1.04,1.08,1.12,1.24,1.04,0.96,0.92,0.84,0.76]
    hr_coef = [1.04,1.08,1.12,1.24,1.04,0.96,0.92,0.84,0.76] # [1.04,1.08,1.12,1.24,1.04,0.96,0.92,0.84,0.76]
    outs = 0
    first_base = ""
    second_base = ""
    third_base = ""
    if num_innings % 9 == 1:
        currBatter = 1
    else:
        currBatter = current_batter
    
    inning_pa = 0
    inning_runners = 0
    while outs < 3:
        prob = random.uniform(0,1)
        result = None
        
        curr_batter_batting_avg_coef = batting_avg_coef[currBatter-1]
        curr_batter_walk_coef = walk_coef[currBatter-1]
        curr_batter_hr_coef = hr_coef[currBatter-1]
        if curr_batter_hr_coef == 888:
            temp_prob = random.uniform(0,1)
            if temp_prob < 0.25:
                result = Result.HOME_RUN
            else:
                result = Result.STRIKE_OUT
        else:
            walk_prob = 0.0741*curr_batter_walk_coef
            hr_prob = 0.031*curr_batter_hr_coef
            single_prob = 0.16*curr_batter_batting_avg_coef
            double_prob = 0.043*curr_batter_batting_avg_coef
            triple_prob = 0.0026*curr_batter_batting_avg_coef
            dp_prob = 0.021
            if prob < walk_prob: 
                result = Result.WALK
            elif prob < walk_prob + hr_prob:
                result = Result.HOME_RUN
            elif prob < walk_prob + hr_prob + single_prob:
                result = Result.SINGLE
            elif prob < walk_prob + hr_prob + single_prob + double_prob:
                result = Result.DOUBLE
            elif prob < walk_prob + hr_prob + single_prob + double_prob + triple_prob:
                result = Result.TRIPLE
            elif prob <  walk_prob + hr_prob + single_prob + double_prob + triple_prob + dp_prob:
                result = Result.DOUBLE_PLAY
            else:
                result = Result.OUT
        #print("current batter is:",currBatter)
        #print("result is:",result)
        if result == Result.WALK:
            if first_base == "Occupied":
                if second_base == "Occupied":
                    if third_base == "Occupied":
                        runs += 1
                    else:
                        third_base = "Occupied"
                else:
                    second_base = "Occupied"
            else:
                first_base = "Occupied"
        
        # Single
        if result == Result.SINGLE:
            # scenarios
            if third_base == "Occupied":
                third_base = "Empty"
                runs += 1
                #print("third_base always scores from single")
            if second_base == "Occupied":
                prob = random.uniform(0,1)
                if prob < 0.6:
                    second_base = "Empty"
                    runs += 1
                    #print("second_base scores from single")
                else:
                    third_base = "Occupied"
                    second_base = "Empty"
                    #print("second_base goes to third_base from single")
                    
            if first_base == "Occupied":
                prob = random.uniform(0,1)
                if prob < 0.65:
                    third_base = "Occupied"
                    # first base goes to third from single
                else:
                    second_base = "Occupied"
            # first base always occupied from single
            first_base = "Occupied"
        
        # Double
        if result == Result.DOUBLE:
            if third_base == "Occupied":
                third_base = "Empty"
                runs += 1
                # third base always scores from double
            if second_base == "Occupied":
                second_base = "Empty"
                runs += 1
                # second base always scores from double
            if first_base == "Occupied":
                prob = random.uniform(0,1)
                if prob < 0.5:
                    third_base = "Occupied"
                    # first base scores half of the time from doubles
                else:
                    runs += 1
                first_base = "Empty"

            second_base = "Occupied"
        
        # Triple
        if result == Result.TRIPLE:
            if first_base == "Occupied":
                first_base = "Empty"
                runs += 1
            if second_base == "Occupied":
                second_base = "Empty"
                runs += 1
            if third_base == "Occupied":
                runs += 1
            # only one left on the base from a triple should be the batter
            third_base = "Occupied"
            #print("TRIPLE")
        
        # Home Run
        if result == Result.HOME_RUN:
            if first_base == "Occupied":
                first_base = "Empty"
                runs += 1
                #print("first_base score from HR")
            if second_base == "Occupied":
                second_base = "Empty"
                runs += 1
                #print("second_base score from HR")
            if third_base == "Occupied":
                third_base = "Empty"
                runs += 1
                #print("third_base score from HR")
            runs += 1
            # batter scores as well
            #print("HOME RUN")
            # no one left on base after home run
        
        if result == Result.OUT:
            prob = random.uniform(0,1)
            if third_base == "Occupied":
                if prob < 0.25:
                    third_base = "Empty"
                    runs += 1
                    # third base can advance to score on fielder's choice about 20% of the times
            if second_base == "Occupied":
                if prob < 0.3 and third_base == "Empty":
                    third_base = "Occupied"
                    second_base = "Empty"
                    # second base can advance to third base 30% of the time on fielder's choice
            if first_base == "Occupied":
                if prob < 0.3 and second_base == "Empty":
                    second_base = "Occupied"
                    first_base = "Empty"
                    # first base can advance to second base 30% of the time on fielder's choice
            outs += 1

        if result == Result.DOUBLE_PLAY:
            if first_base == "Occupied":
                first_base = "Empty"        
            outs += 2
        
        if result == Result.STRIKE_OUT:
            outs += 1

        currBatter = ((currBatter % 9) + 1)
    #print("Runs is:",runs)
    return currBatter,runs


total_runners = 0
plate_apperances = 0
total_runs_scored = 0
num_innings = 900000
batter = 1
for i in range(num_innings):
    next_batter, i_runs_scored = inning(batter,i)
    batter = next_batter
    total_runs_scored += i_runs_scored
average_runs_scored = round(total_runs_scored/(num_innings/9),3)
print("Average Runs Scored Per Game:",average_runs_scored)

