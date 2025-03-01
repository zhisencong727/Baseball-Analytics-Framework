import csv
x = []
y = []
with open("Phillies_Data.csv","r") as file:
    reader = csv.reader(file)
    next(reader, None)
    for row in reader:
        temp = []
        for i in range(len(row)-1,4,-1):
            if row[i] != '':
                temp.append(round(float(row[i]),3))
        if len(temp) > 2 and temp[len(temp)-2] != 0.0:
            x.append(temp[0:len(temp)-2])
            y.append(temp[len(temp)-2])

import numpy as np
from sklearn.ensemble import RandomForestRegressor

max_length = max(len(lst) for lst in x)+2
X_padded = np.array([lst + [0]*(max_length - len(lst)) for lst in x])
y = np.array(y)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_padded, y)

x_16_to_20 = []
y_21 = []
with open("Phillies_Data.csv","r") as file:
    reader = csv.reader(file)
    next(reader, None)
    for row in reader:
        temp = []
        for i in range(len(row)-1,4,-1):
            if row[i] != '':
                temp.append(round(float(row[i]),3))
        if len(temp) > 2 and temp[len(temp)-2] != 0.0:
            x_16_to_20.append(temp)
            y_21.append(round(float(row[4]),3))


max_length_new = max(len(lst) for lst in x_16_to_20)
X_padded_new = np.array([lst + [0]*(max_length_new - len(lst)) for lst in x_16_to_20])

y_pred = model.predict(X_padded_new)
print("Predicted vs Actual values:")
for idx, (pred, actual) in enumerate(zip(y_pred, y_21)):
    print(f"Sample {idx + 1}: Predicted = {pred:.2f}, Actual = {actual}")