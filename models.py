import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
import pickle 


with open('df_eda.bin', 'rb') as f:
    onehotencoder, X, Y = pickle.load(f)



model = Ridge(alpha=1.0)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

#parâmetros para previsões: Engine Cylinder, Bore (mm), Stroke (mm), Cooling System
model.fit(X, Y)


aux = np.random.randint(1, 10000)
moto1 = X[aux]
moto1_pred = model.predict([moto1])

print(f'Moto escolhida: {aux}')
print(f"Potência moto 1 (prediction): {moto1_pred}")
print(f"Potência moto 1 (real): {Y[aux]}")


with open('df_model.bin', 'wb') as f:
    pickle.dump((onehotencoder, model, X, Y), f)
