from sklearn import svm
from sklearn.metrics import accuracy_score
import level4
 
# データを学習用とテスト用に分割する
x_train_data,x_test,y_train_data,y_test = level4.load_linear_ex()
 
# データを学習（LinearSVCモデルを選択）
clf = svm.LinearSVC() 
clf.fit(x_train_data, y_train_data)
 
# 予測して精度を確認する
y_predict = clf.predict(x_test) 
X = accuracy_score(y_test, y_predict) * 100
print(X)
