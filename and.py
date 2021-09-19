from utils.model import Perceptron
from utils.all_utils import data,save_model,save_plot
import pandas as pd
import numpy as np
import logging
import os

logging_str="[%(asctime)s:%(levelname)s:%(module)s] %(message)s"
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,'running_log.log'),level=logging.INFO,format=logging_str,filemode='a')
def main(data1,lr,epochs):
    
    df_AND=pd.DataFrame(data1)
    print(df_AND)

    X,y=data(df_AND)

    lr=0.3
    epochs=10

    model_AND=Perceptron(lr=lr,epochs=epochs)
    model_AND.fit(X,y)
    _=model_AND.total_loss()

    save_model(model_AND,filename="AND.model")
    save_plot(df_AND,'AND.png',model_AND)

if __name__=="__main__":
    AND={
        'X1':[0,0,1,1],
        'X2':[0,1,0,1],
        'y':[0,0,0,1]
        }
    lr=0.3
    epochs=10
    main(data1=AND,lr=lr,epochs=epochs)