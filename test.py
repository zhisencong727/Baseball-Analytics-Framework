x = [[1],[2],[3],[4],[5],[6],[7]]
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
print(scaler_x.fit_transform(x))
x_2016 = [[10],[20],[30],[40],[50],[60],[70]]
print(scaler_x.transform(x_2016))



