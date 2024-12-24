import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("alzheimers_disease_data.csv", index_col=0)

print(df.head())

from sklearn.preprocessing import StandardScaler

# 'DoctorInCharge' sütununu veri setinden çıkarın
df = df.drop('DoctorInCharge', axis=1)

scaler = StandardScaler()
scaler.fit(df.drop('Diagnosis', axis=1))
scaled_features = scaler.transform(df.drop('Diagnosis', axis=1))

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
# print(df_feat.head())

# Veri setini eğitim ve test setlerine ayırın
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['Diagnosis'], test_size=0.30)

# KNN modelini eğitin
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

# Modelin değerlendirilmesi
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# K değerini belirlemek için bir hata oranı grafiği oluşturun
error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='black', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# K = 1 ile başlayıp denemeler yapalım
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('K=1')
print('\n')

# K=24
knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(' K=24')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))