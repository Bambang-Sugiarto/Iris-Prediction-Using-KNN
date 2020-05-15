import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

bunga = datasets.load_iris()
print(bunga)
print("")
print("-> Target Names")
print(bunga.target_names)
x=bunga.data
y=bunga.target
df = pd.DataFrame(x, columns=bunga.feature_names)
print("")
print(df.head())
print("")
knn = KNeighborsClassifier(n_neighbors=6,weights='uniform',algorithm='auto',metric='euclidean')
x_train = bunga['data']
y_train = bunga['target']
knn.fit(x_train, y_train)

Data = [[6.2, 1.5, 4.2, 2.6]]

y_pred = knn.predict(Data)
print("")
print("-> Prediksi : ",Data)
print("-> Note : 0 = Setosa, 1 = Versicolor, 2 = Virginica")
print("-> Hasil Prediksi : Jenis Bunga ",y_pred)
