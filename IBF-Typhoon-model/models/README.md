# Models folder

This folder contains the different models used and the results obtained. There are three model types: Binary Classification, Multiclass Classification and Regression. The folders with the corresponding names contain a file for each trained and tested model type. 

The estimated model performance for each model is obtained in two steps. 

1. The full data set is used to perform feature selection and to find the optimal hyperparameters.

2. With the selected features, Nested Cross validation is used to obtain the performance estimate. *Elaborate on splitting train and test

**Output folder** <br>
This folder contains the outputs of the different models. The predictions for the test set to obtain the performance score and the selected hyperparameters

**Ultility functions folder** <br>
This folder contains all the utility functions used

**Model Results** <br>
This jupyter notebook contains the model and performance estimations. The scores are compared and the optimal models obtained.

**Saved models folder** <br>
Contains the trained optimal models (on the full dataset)

