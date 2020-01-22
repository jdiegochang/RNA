import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import numpy as np
#-------------------------------------------Find Best Parameters-----------------------------------------#
def SeachGridCrossValidation():
    param_grid = {'hidden_layer_sizes': [i for i in range(1,5)],'activation': ['tanh','logistic'],
              'learning_rate_init': [a/1000 for a in range(1,5)]}
    mlp = MLPRegressor()
    GS = GridSearchCV(mlp, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5, verbose=True)
    GS.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(GS.best_params_)
    print()
    return(GS.best_params_)
#-------------------------------------------Data Fetch Function------------------------------------------#
def readData(file):
    df = pd.read_csv(file, dtype=float)
    return df
##########################------------------------MAIN------------------##################################
if __name__ == "__main__":
# ----------------------------------------------Data Fetch-----------------------------------------------#
    cData=readData('BasedeDatos.csv')
    df_x=cData.iloc[:,0:48]
    df_y=cData.iloc[:,48:]
# ---------------------------------------Split in Training and Testing-----------------------------------#
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.10,random_state=21)
# ---------------------------------------------MLP Creation----------------------------------------------#
    error=[]
    best_params=SeachGridCrossValidation()
