import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
plt.style.use("fivethirtyeight")

def data(df):
  """It is used to seperate the dependent and idependent features

  Args:
      df (pd.Dataframe): Tables

  Returns:
      tuple it returns the tuples o dependent fetaure
  """
  X=df.drop('y',axis=1)
  y=df['y']
  return X,y

def save_model(model,filename):
  model_dir='models1'
  os.makedirs(model_dir,exist_ok=True)
  filePath=os.path.join(model_dir,filename )
  joblib.dump(model,filePath)
  print(filePath)



def save_plot(df,file_name,model):
  def _create_base_plot(df):
    df.plot(kind='scatter',x='X1',y='X2',c='y',s=100,cmap='winter')
    plt.axhline(y=0,color='black',linestyle="--",linewidth=1)
    plt.axvline(x=0,color='black',linestyle="--",linewidth=1)
    figure=plt.gcf()
    figure.set_size_inches(10,8)

  def _plot_decision_region(X,y,classifier,resolution=0.02):
    colors=("red","blue","lightgreen","gray","cyan")
    cmap=ListedColormap(colors[:len(np.unique(y))])
    X=X.values
    x1min,x1max=X[:,0].min()-1,X[:,0].max()+1
    x2min,x2max=X[:,1].min()-1,X[:,1].max()+1
    print(x1min,x1max)
    print(x2min,x2max)
    xx1,xx2=np.meshgrid(np.arange(x1min,x1max,resolution),np.arange(x2min,x2max,resolution))
    print(xx1,xx2)
    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.2,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    plt.plot()


  X,y=data(df)
  _create_base_plot(df)
  _plot_decision_region(X,y,model)
  plot_dir='plots'
  os.makedirs(plot_dir,exist_ok=True)
  plotPath=os.path.join(plot_dir,file_name)
  plt.savefig(plotPath)