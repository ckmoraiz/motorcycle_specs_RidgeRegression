import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

with open('df_model.bin', 'rb') as f:
    onehotencoder, model, X, Y = pickle.load(f)


print("Insert bore value in mm:")
bore = input(float)

print("Insert stroke value in mm:")
stroke = input(float)

print("Choose the number o cylinder configuration:")
print("[0]Diesel                       [1]Four cylinder boxer         [2]In-line four                [3]In-line six             [4]In-line three\n\
[5]Single cylinder              [6]Six cylinder boxer          [7]Square four cylinder        [8]Twin                    [9]Two cylinder boxer\n\
[10]V2                          [11]V3                         [12]V4                         [13]V6                     [14]V8")

cylinder = input(int)


print("Choose the number of the cooling system type:")
print("[0]Air      [1]Liquid      [2]Oil & Air")
cooling = input(int)



X = [[cylinder, bore, stroke, cooling]]


# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# X = X.reshape( -1, 4)
X = pd.DataFrame(X, columns=['Engine cylinder', 'Bore (mm)', 'Stroke (mm)', 'Cooling system'])

X = onehotencoder.transform(X)


moto1_pred = model.predict(X)


print(f"Potência final da moto com as especificações escolhidas: {moto1_pred}hp")


