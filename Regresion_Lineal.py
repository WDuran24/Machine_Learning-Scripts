
#REGRESION LINEAL 

import pandas as pd
import matplotlib.pyplot as plt




data=pd.read_csv('/content/Regresion_lineal_data.csv',sep=';',encoding='latin-1')
data.head()

data.isnull().sum()# para descartar valores nulos

data.describe()

data.hist(figsize=(14,12))
plt.show

data.drop(data[data.Precio>2000000].index,inplace=True)# para filtrar filas

data.hist(figsize=(14,12))
plt.show

data['Precio'].hist(figsize=(14,12))
plt.show

#boxplots

data.plot(kind='box',subplots=True,layout=(5,2),figsize=(20,20),sharex=False)
plt.show()

#Scatterplot
plt.scatter(data.Metros,data.Precio,color='darkblue',s=2)
plt.title('Scatterplot')
plt.xlabel('Metros')
plt.ylabel('Precio')
plt.show()

import seaborn as sns

#Grafico de Correlacion

corr_data=data[['Precio', 'Dormitorios', 'Baños', 'Pisos', 'Oceanview', 'Metros',
       'Año_Fabricacion']].corr(method='pearson')
plt.figure(figsize=(8,6))
sns.heatmap(corr_data,annot=True)
plt.show()

data=pd.get_dummies(data)
data.tail()

#Regresion con statsmodels

import statsmodels.formula.api as smf

regresion_simple=smf.ols(formula='Precio~Metros',data=data).fit()
regresion_simple.summary()
#y=2251+143.6306

pred=regresion_simple.predict(data['Metros'])

plt.scatter(data.Metros,data.Precio,color='darkblue',s=2)
plt.scatter(data.Metros,pred,color='orange',s=2)
plt.title('Regresion Simple')
plt.xlabel('Metros')
plt.ylabel('Precio')
plt.show()

#Regresion Multiple

regresion_mul=smf.ols(formula='Precio~Metros+Baños+Dormitorios+Pisos+Año_Fabricacion'
,data=data).fit()

regresion_mul.summary()



#Regresion lineal con sklearn

from sklearn.feature_selection import RFE #Nos ayuda con la seleccion de variables

#Para la estimacion
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

data.columns

feature_cols=['Dormitorios', 'Baños', 'Pisos', 'Oceanview', 'Metros',
       'Año_Fabricacion', 'Año_Renovacion', 'Latitud', 'Longitud', 'Tipo_Casa',
       'Tipo_Departamento']

x=data[feature_cols]
y=data['Precio']

estimator=SVR(kernel='linear')

selector=RFE(estimator,n_features_to_select=2,step=1)
selector=selector.fit(x,y)

selector.support_

list(zip(feature_cols,selector.support_))

selector.ranking_

x_pred=x[['Tipo_Casa','Tipo_Departamento']]

Rl2=LinearRegression()
Rl2.fit(x_pred,y)


Rl2.intercept_

Rl2.coef_

list(zip(x_pred,Rl2.coef_))

list(zip(x_pred,Rl2.coef_))

Rl2.score(x_pred,y)