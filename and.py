from utils.model import Perceptron
from utils.all_utils import data,save_model,save_plot
import pandas as pd
import numpy as np


AND={
    'X1':[0,0,1,1],
    'X2':[0,1,0,1],
    'y':[0,0,0,1]
}
df_AND=pd.DataFrame(AND)
print(df_AND)

X,y=data(df_AND)

lr=0.3
epochs=10

model_AND=Perceptron(lr=lr,epochs=epochs)
model_AND.fit(X,y)

save_model(model_AND,filename="AND.model")
save_plot(df_AND,'AND.png',model_AND)
