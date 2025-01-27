from pitcher import pitcher
import csv
#pitcher1 = pitcher("clayton","kershaw")

def get_data(augment_length_one):

    pitch_encoding_map = {
        '4-Seam Fastball': 0.0,
        '2-Seam Fastball': 1.0,
        'Cutter': 2.0,
        'Slider': 3.0,
        'Curveball': 4.0,
        'Changeup': 5.0,
        'Knuckleball': 6.0,
        'Sinker': 7.0,
        'Slurve': 8.0,
        'Knuckle Curve': 9.0,
        'Splitter': 10.0,
        'Sweeper': 11.0
    }
    description_encoding_map = {
        'foul' : 0.0,
        'ball' : 1.0,
        'blocked_ball' : 1.0,
        'called_strike' : 2.0,
        'swinging_strike' : 3.0,
        'swinging_strike_blocked' : 3.0,
        'missed_bunt' : 4.0,
        'foul_bunt' : 4.0,
        'hit_into_play' : 5.0,
        'hit_by_pitch' : 6.0,
        'foul_tip' : 7.0
    }

    team = "LAD"

    currAtBat = []
    allAtBats = []
    i = 0
    pitch_type_set = set()
    description_set = set()
    with open("pitchLogFiles/clayton_kershaw_pitch_log.csv",'r') as csvfile:
        reader =csv.DictReader(csvfile)
        for row in reader:
            if row['pitch_type'] == "" or row['pitch_name'] == "Intentional Ball" or row['pitch_name'] == "Other":
                continue
            if row['pitch_name'] == "Split-Finger":
                row['pitch_name'] = "Splitter"
            pitch_type_set.add(row['pitch_name'])
            description_set.add(row['description'])
            if row['events'] != "":
                currAtBat.reverse()
                allAtBats.append(currAtBat)
                currAtBat = []
            
            pitch_type = row['pitch_name']
            pitch_type = pitch_encoding_map[pitch_type]
            outs = float(row['outs_when_up'])
            ball = float(row['balls'])
            strike = float(row['strikes'])
            batter_hand = row['stand']
            if batter_hand == "L":
                batter_hand = -1.0
            else:
                batter_hand = 1.0
            inning = float(row['inning'])
            velocity = float(row['release_speed'])
            runner_on_base = 0.0
            if row['on_1b'] != "":
                runner_on_base += 1.0
            if row['on_2b'] != "":
                runner_on_base += 2.0
            if row['on_3b'] != "":
                runner_on_base += 3.0
            if team == row['home_team']:
                score_diff = float(row['home_score'])-float(row['away_score'])
            else:
                score_diff = float(row['away_score'])-float(row['home_score'])
            description = description_encoding_map[row['description']]

            # pitch_type,outs,ball,strike,batter_hand,inning,velocity,runner_on_base,score_diff,description
            currPitchInfo = [pitch_type,outs,ball,strike,batter_hand,inning,velocity,runner_on_base,score_diff,description]
            currAtBat.append(currPitchInfo)


    print("pitch_type,outs,ball,strike,batter_hand,inning,velocity,runner_on_base,score_diff,description")



    k = 0
    while k < len(allAtBats):
        each = allAtBats[k]
        if len(each) < 2:
            allAtBats.remove(each)
        else:
            k += 1

    print(pitch_type_set)
    print("length of allAtBats is:",len(allAtBats))
    max_length = max(len(seq) for seq in allAtBats)
    print("max_length is: ",max_length)

    def pad_sequences(each_sequence,max_length):
        padding = [0.0]*10
        while len(each_sequence) < max_length:
            each_sequence.append(padding)
        return each_sequence


    def extractOutput(seq):
        for i in range(len(seq)-1,0,-1):
            if sum(seq[i]) != 0.0:
                output = seq[i][0]
                seq[i] = [0.0]*10
                return seq,output

    if augment_length_one:
        training_data = []
        targets = []
        for each_at_bat in allAtBats:
            for i in range(len(each_at_bat)-1):
                training_data.append(each_at_bat[i])
                targets.append(each_at_bat[i+1][0])
        #print(training_data[0:10])
        #print(targets[0:10])
        #print(allAtBats[0:1])
        return training_data,targets
    
    else:    
        print("THIS IS TRIGGERED")
        for each_sequence in allAtBats:
            each_sequence = pad_sequences(each_sequence, max_length)

        for i in range(len(allAtBats)):
            allAtBats[i],output = extractOutput(allAtBats[i])
            allAtBats[i] = [allAtBats[i],output]

        print(description_set)
        print(allAtBats[0:1])
        return allAtBats,max_length

    


#get_data(False)