import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# -------------------------------------------Learning Curve--------------------------------------------#
def LCPlot(mse):
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.set_ylabel("MSE",fontsize=16)
    ax.get_yaxis().tick_left()
    ax.set_xlabel("Epochs",fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.plot(mse,lw=2.5)
    plt.show()
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
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.15,random_state=42)
# ---------------------------------------------MLP Creation----------------------------------------------#

    v_lr=0.1
    v_act="logistic"
#   "tanh","identity","relu"
    v_hls=32
    mlp = MLPRegressor(hidden_layer_sizes=(v_hls), activation= v_act, learning_rate_init=v_lr, verbose=True)
# ---------------------------------------------MLP Training-----------------------------------------------#
    mlp.fit(X_train,y_train)
# ----------------------------------------------MLP Testing-----------------------------------------------#
    y_predictions = mlp.predict(X_test)
# ---------------------------------------------Learning Curve---------------------------------------------#
    print(X_test[:1])
    print(y_predictions[0])
    #LCPlot(mlp.loss_curve_)
#---------------------------------------------------------------------------------------------------------#
