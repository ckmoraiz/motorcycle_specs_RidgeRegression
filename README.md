# Motorcycle Specs Classification with Ridge Regression
In this project I used a dataset with engine specification of different motorcycle models.

The goal was to create a prediction application capable of predict the final power (hp) of the engine given some features. 

The features chosen was: Bore (mm), Stroke (mm), Engine Configuration (4 cylinder in-line, for example) and the cooling system. 


The model chosen was the Ridge Regression with cross validation (10 k-folds). 
The metric of evaluation chosen was the mean absolute error (MAE). 
The results are ilustrated below:

![alt text](https://github.com/chydrue/motorcycle_specs_NaiveBayes/blob/main/prediction.png)


P.S.: The final app.py script need further debugging process, since the implementation of encoding in the input of new data failed. 
This happenned because the encoder was defined in another script, in a way that the input data it's not processed in the proper way. 

This can be corrected by importing the same encoder and arranging the data structure properly. This will be done in the future. Maybe. Probably not.
