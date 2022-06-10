import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import warnings

warnings.warn("msg", FutureWarning)

sns.set(color_codes=True)

df = pd.read_csv('Output.csv')
# To display the top 5 rows
print(df.head(5))
print(df.tail(5))

plt.figure(figsize=(20,10), dpi=300)
plt.title("Correlation")
c= df.corr()
sns.heatmap(c,annot=True, linewidths=.5)
# plt.savefig("Correlation.png")



X = df['Exact Solution cost'].values.reshape(-1, 1)
Y = df['Heuristic Solution cost'].values.reshape(-1, 1)
NODE = df['node'].values.reshape(-1, 1)

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
fig = plt.figure(figsize=(20,10), dpi=300)
plt.scatter(X, Y, color='indigo', alpha=0.5, s = 100)
plt.plot(X, Y_pred, color='lightcoral', linewidth=3)
plt.title("Linear Regression (Exact Solution cost VS Heuristic Solution cost")
plt.xlabel("Exact Solution cost")
plt.ylabel("Heuristic Solution cost")
plt.margins(x=0.05, y=0.05, tight=True)
plt.show()
# plt.savefig("Linear Regression (Exact Solution cost VS Heuristic Solution cost.png")



X1 = df['Exact Solution computation'].values.reshape(-1, 1)
Y1 = df['Heuristic Solution computation'].values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(X1, Y1)
Y_pred = linear_regressor.predict(X1)
fig = plt.figure(figsize=(20,10), dpi=300)
plt.scatter(X1, Y1, color='indigo', alpha=0.5, s = 100)
plt.plot(X1, Y_pred, color='lightcoral', linewidth=3)
plt.title("Linear Regression (Exact Solution computation VS Heuristic Solution computation")
plt.xlabel("Exact Solution computation")
plt.ylabel("Heuristic Solution computation")
plt.margins(x=0.05, y=0.05, tight=True)
plt.show()
# plt.savefig("Linear Regression (Exact Solution computation VS Heuristic Solution computation.png")


# Histogram section
kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

plt.figure(figsize=(20,15), dpi= 300)
sns.distplot(df['Exact Solution computation'], color="deeppink", bins = 10, **kwargs)
plt.title("Histogram (Exact Solution computation)")
# plt.savefig("Histogram (Exact Solution computation).png")


plt.figure(figsize=(20,10), dpi= 300)
sns.distplot(df['Heuristic Solution computation'], color="orange", bins = 10, **kwargs)
plt.title("Histogram (Heuristic Solution computation)")
plt.savefig("Histogram (Heuristic Solution computation).png")

plt.figure(figsize=(20,10), dpi= 300)
sns.distplot(df['Exact Solution cost'], color="deeppink", bins = 10, **kwargs)
plt.title("Histogram (Exact Solution cost)")
# plt.savefig("Histogram (Exact Solution cost).png")


plt.figure(figsize=(20,10), dpi= 300)
sns.distplot(df['Heuristic Solution cost'], color="orange", bins = 10, **kwargs)
plt.title("Histogram (Heuristic Solution cost)")
# plt.savefig("Histogram (Heuristic Solution cost).png")

# line chart
# plt.figure(figsize=(20,15), dpi= 300)
# line_chart1 = plt.plot(X1, color='lightcoral', marker='o', linestyle='dashed', linewidth=5, markersize=3)
# plt.title('Line chart (Exact Solution computation)')
# plt.xlabel('Nodes')
# plt.ylabel('Computation time')
# plt.show()


# plt.figure(figsize=(20,15), dpi= 300)
# line_chart1 = plt.plot(Y1, color='indigo', marker='o', linestyle='dashed', linewidth=5, markersize=3)
# plt.title('Line chart (Heuristic Solution computation)')
# plt.xlabel('Nodes')
# plt.ylabel('Computation time')
# plt.show()


# plt.figure(figsize=(20,15), dpi= 300)
# line_chart1 = plt.plot(X, color='lightcoral', marker='o', linestyle='dashed', linewidth=5, markersize=3)
# plt.title('Line chart (Exact Solution cost)')
# plt.xlabel('Nodes')
# plt.ylabel('Cost')
# plt.show()

# plt.figure(figsize=(20,15), dpi= 300)
# line_chart1 = plt.plot(Y, color='indigo', marker='o', linestyle='dashed', linewidth=5, markersize=3)
# plt.title('Line chart (Heuristic Solution cost)')
# plt.xlabel('Nodes')
# plt.ylabel('Cost')
# plt.show()

































