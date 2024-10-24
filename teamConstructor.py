from pitcher import pitcher
from batter import batter

class team:
    def __init__(self,teamname,lineup,sp):
        self.teamname = teamname
        self.lineup = lineup
        self.sp = sp

def teamConstructor(day_night,awayTxt,homeTxt,awayPitcherHand,homePitcherHand,awayName,homeName):
    
    awaySPvsL = None
    awaySPvsR = None
    awaySPArr = []
    awayLineup = []
    
    with open(awayTxt,"r") as awayFile:
        i = 0
        for line in awayFile:
            line = line.split(" ")
            line[2] = line[2][1]
            if i == 0:
                if len(line) == 5:
                    awaySPvsL = pitcher(line[0],line[1],"L",int(line[4].strip()),day_night,"away")
                    awaySPvsR = pitcher(line[0],line[1],"R",int(line[4].strip()),day_night,"away")
                else:
                    awaySPvsL = pitcher(line[0],line[1],"L",0,day_night,"away")
                    awaySPvsR = pitcher(line[0],line[1],"R",0,day_night,"away")
            else:
                if len(line) == 5:
                    tempBatter = batter(line[0],line[1],homePitcherHand,int(line[4].strip()),day_night,"away")
                else:
                    tempBatter = batter(line[0],line[1],homePitcherHand,0,day_night,"away")
                #print(tempBatter.lastname)
                awayLineup.append(tempBatter)
            i += 1
            
    homeSPvsL = None
    homeSPvsR = None
    homeSPArr = []
    homeLineup = []
    
    with open(homeTxt,"r") as homeFile:
        i = 0
        for line in homeFile:
            line = line.split(" ")
            #print(line)
            if i == 0:
                if len(line) == 5:
                    homeSPvsL = pitcher(line[0],line[1],"L",int(line[4].strip()),day_night,"home")
                    homeSPvsR = pitcher(line[0],line[1],"R",int(line[3].strip()),day_night,"home")
                else:
                    homeSPvsL = pitcher(line[0],line[1],"L",0,day_night,"home")
                    homeSPvsR = pitcher(line[0],line[1],"R",0,day_night,"home")
            else:
                if len(line) == 5:
                    #print(line)
                    tempBatter = batter(line[0],line[1],awayPitcherHand,int(line[4].strip()),day_night,"home")
                else:
                    tempBatter = batter(line[0],line[1],awayPitcherHand,0,day_night,"home")
                #print(tempBatter.lastname)
                homeLineup.append(tempBatter)
            i += 1
    
    
    with open(homeTxt,"r") as homeFile2:
        i = 0
        for line in homeFile2:
            line = line.split(" ")
            line[2] = line[2][1]
            if i > 0:
                if line[2] == "L\n" or line[2] == "L":
                    #print(line[1])
                    awaySPArr.append(awaySPvsL)
                elif line[2] == "R\n" or line[2] == "R":
                    #print(line[1])
                    awaySPArr.append(awaySPvsR)
                elif line[2] == "S\n" or line[2] == "S":
                    if awayPitcherHand == "L":
                        awaySPArr.append(awaySPvsR)
                    elif awayPitcherHand == "R":
                        awaySPArr.append(awaySPvsL)
            i += 1
    
    with open(awayTxt,"r") as awayFile2:
        i = 0
        for line in awayFile2:
            line = line.split(" ")
            line[2] = line[2][1]
            #print(line)
            if i > 0:
                if line[2] == "L\n" or line[2] == "L":
                    homeSPArr.append(homeSPvsL)
                    #print(line[1] + "L")
                elif line[2] == "R\n" or line[2] == "R":
                    homeSPArr.append(homeSPvsR)
                    #print(line[1] + "R")
                elif line[2] == "S\n" or line[2] == "S":
                    if homePitcherHand == "L":
                        homeSPArr.append(homeSPvsR)
                    elif homePitcherHand == "R":
                        homeSPArr.append(homeSPvsL)
                
            i += 1
        #print("i is: ",str(i))
        
    
    awayTeamObject = team(awayName,awayLineup,awaySPArr)
    homeTeamObject = team(homeName,homeLineup,homeSPArr)
    
    return awayTeamObject,homeTeamObject