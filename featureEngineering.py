import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


pd.options.display.width = 0

df = pd.read_csv('Dados/df_clean.csv')
df.drop(columns='Unnamed: 0', inplace=True)


#Variáveis categóricas = Engine cylinder, Cooling system, Transmission type

encoder = LabelEncoder()
scaler = MinMaxScaler()


df['Engine cylinder'] = encoder.fit_transform(df['Engine cylinder'])
df['Cooling system'] = encoder.fit_transform(df['Cooling system'])
df['Transmission type'] = encoder.fit_transform(df['Transmission type'])
df.drop(df[df['Stroke (mm)'] == '1,093.0'].index, inplace=True)


X = df.iloc[:, 2:6]
Y = df.iloc[:, 1]
# print(X.head(5))
# X.drop(X[X['Stroke (mm)'] == '1,093.0'].index, inplace=True)
# df['Stroke (mm)'] = df['Stroke (mm)'].astype(float)
# print(X.dtypes)
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=['Engine cylinder', 'Bore (mm)', 'Stroke (mm)', 'Cooling system'])


onehotencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 2, 3])], remainder='passthrough') 
X = onehotencoder.fit_transform(X).toarray()

with open('df_eda.bin', 'wb') as f:
    pickle.dump((onehotencoder, X, Y), f)






# sns.pairplot(df.sample(100))
# plt.show(block=True)

