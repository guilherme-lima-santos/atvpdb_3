import numpy as pd 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv ('dados.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.85,random_state=7)


linearRegression = LinearRegression()
linearRegression.fit(x_treinamento, y_treinamento)


y_pred = linearRegression.fit(x_treinamento, y_treinamento)


plt.scatter(x_treinamento, y_treinamento, color="red")
plt.plot(x_treinamento,
linearRegression.predict(x_treinamento),
color="blue")
plt.title("Salário x Tempo de Experiência (Treinamento)")
plt.xlabel("Anos de Experiência")
plt.ylabel("Salário")
plt.show()



plt.scatter(x_teste, y_teste, color="red")
#os coeficientes são únicos, assim não faz diferença
#trocar as coleções na hora de exibir
plt.plot(x_treinamento,
linearRegression.predict(x_treinamento),
color="blue")
plt.title("Salário x Tempo de Experiência (Teste)")
plt.xlabel("Anos de Experiência")
plt.ylabel("Salário")
plt.show()


print(linearRegression.predict([[15.7]]))


print(linearRegression.predict([[0]]))


print(f'y = {linearRegression.coef_[0]:.2f}x +{linearRegression.intercept_:.2f}')