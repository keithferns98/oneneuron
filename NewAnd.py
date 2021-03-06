from utils.model import Perceptron
from utils.all_utils import data,save_model,save_plot
import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm

logging_str="[%(asctime)s:%(levelname)s:%(module)s] %(message)s"
log_dir="logs1"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,'running_log.log'),level=logging.INFO,format=logging_str,filemode="a")
#logging.basicConfig(level=logging.INFO,format=logging_str)
def main(data1,lr,epochs):
    
    df_AND=pd.DataFrame(data1)
    logging.info(f"This is actual data {df_AND}")

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
    epochs=100
    try:
        logging.info(">>>>>>>>> starting training >>>>>>>>>>")
        main(data1=AND,lr=lr,epochs=epochs)
        logging.info(">>>>>>>> ending training >>>>>>>>>>")
    except Exception as e:
        logging.exception(e) 
        raise e 