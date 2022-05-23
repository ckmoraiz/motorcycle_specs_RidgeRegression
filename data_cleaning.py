import pandas as pd
pd.options.display.width = 0

df = pd.read_csv('Dados/all_bikez_curated.csv')
# print(df.columns)
df_rating = df.loc[:, ['Power (hp)', 'Engine cylinder', 'Bore (mm)', 'Stroke (mm)', 'Cooling system', 'Transmission type']]
df = df_rating.dropna(inplace=True)
print(df_rating['Engine cylinder'].value_counts())

df_rating.to_csv('Dados/df_clean.csv')