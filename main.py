import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pandas.read_csv('cost_revenue.csv')

X = DataFrame(data, columns=['production_budget_usd'])
Y = DataFrame(data, columns=['worldwide_gross_usd'])

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.xlim(0, 450000000)
plt.ylim(0, 3000000000)

regression = linear_model.LinearRegression()
regression.fit(X, Y)
plt.plot(X, regression.predict(X), color='red', linewidth=3)

plt.show()






