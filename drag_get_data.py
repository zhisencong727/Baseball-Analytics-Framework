import csv 

def get_data(year):
    x = []
    y = []
    with open("astros_drag.csv","r") as file:
        reader =csv.DictReader(file)
        for row in reader:
            if "NA" in row.values():
                continue
            if float(row["hit_vertical_angle"]) < 00:
                continue
            if float(row["hit_distance"]) < 0:
                continue
            if float(row["hit_vertical_angle"]) > 80:
                continue


            # 0 for right handed pitcher, 1 for left handed pitcher
            if row["pitcher_throws"] == "R":
                row["pitcher_throws"] = 0.0
            else:
                row["pitcher_throws"] = 1.0

            # 0 for right handed batter, 1 for left handed batter    
            if row["bat_side"] == "R":
                row["bat_side"] = 0.0
            else:
                row["bat_side"] = 1.0

            # 0 for FF (4-seamer), 1 for FT (2-seamer)
            if row["pitch_type"] == "FF":
                row["pitch_type"] = 0.0
            else:
                row["pitch_type"] = 1.0
            
            if row["year"] in year:
                pitcher_hand = row["pitcher_throws"]
                batter_hand = row["bat_side"]
                pitch_type = row["pitch_type"]
                release_speed = round(float(row["release_speed"]),3)
                plate_speed = round(float(row["plate_speed"]),3)
                hit_exit_speed = round(float(row["hit_exit_speed"]),3)
                hit_spinrate = round(float(row["hit_spinrate"]),3)
                hit_vertical_angle = round(float(row["hit_vertical_angle"]),3)
                hit_bearing = round(float(row["hit_bearing"]),3)
                x.append([pitcher_hand,batter_hand,pitch_type,release_speed,plate_speed,hit_exit_speed,hit_spinrate,hit_vertical_angle])
                y.append(round(float(row["hit_distance"]),3))
    return x,y