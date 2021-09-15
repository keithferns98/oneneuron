from utils.model import Perceptron
from utils.all_utils import data,save_model,save_plot
import pandas as pd
import numpy as np


OR={
    'X1':[0,0,1,1],
    'X2':[0,1,0,1],
    'y':[0,1,1,1]
}
df_OR=pd.DataFrame(OR)
print(df_OR)

X,y=data(df_OR)

lr=0.3
epochs=10

model_OR=Perceptron(lr=lr,epochs=epochs)
model_OR.fit(X,y)

save_model(model_OR,filename="AND.model")
save_plot(df_OR,'OR.png',model_OR)
