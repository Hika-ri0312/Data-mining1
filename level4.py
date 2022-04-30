from sklearn.model_selection import train_test_split
import pandas as pd

def load_linear_ex():
    df = pd.read_csv("winequality-white.csv",sep=";",encoding="utf-8")
    
    y = df['quality']
    x = df.drop(columns='quality') #'quality'カラムを除く。
    
    # データを学習用とテスト用に分割する
    x_train_data,x_test,y_train_data,y_test = train_test_split(x,y,test_size=0.2)

    return x_train_data,x_test,y_train_data,y_test