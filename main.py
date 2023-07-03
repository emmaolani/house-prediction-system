import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("train.csv")
data.drop(data.columns[[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                        63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                        78, 79]], axis=1, inplace=True)
# print(data.head(1))
# initializing the datastructure
size_weight = data['LotArea'].to_numpy().reshape(-1, 1)
price_weight = data['SalePrice'].to_numpy().reshape(-1, 1)

# print(len(size_weight))
# print(len(price_weight))

# grid search over all possible theta values and compute
start, end, steps = -20, 20, 5
theta_0, theta_1 = np.arange(start, end, steps), np.arange(start, end, steps)


def err_cost(predictions, target, l):
    N = predictions.shape[0]
    diff = predictions.ravel() - target.ravel()
    print(diff.T)
    cost = np.dot(diff, diff.T) / N
    return cost


def linearModel(thet, x, l):
    x = np.hstack((x, np.ones((x.shape[0], 1), dtype=x.dtype)))
    return np.dot(x, thet)


# loop over all the parameter pairs and create a list of all possible pairs
theta_list = []
l = 0
for first in theta_0:
    for second in theta_1:
        li = np.array([first, second]).reshape(-1, 1)
        theta_list.append(li)

linear_cost = []
for thetas in theta_list:
    pred_linear = linearModel(thetas, size_weight, l)
    linear_cost.append(err_cost(pred_linear, price_weight, l))
    l = l + 1


axis_length = len(np.arange(start, end, steps))
linear_cost_matrix = np.array(linear_cost).reshape(axis_length, axis_length)
#
# print(size_weight.ravel())
# print(price_weight.ravel())
plt.scatter(size_weight.ravel(), price_weight.ravel(), c="blue")
plt.show()

fig = go.Figure(data=go.Surface(z=linear_cost_matrix,
                                x=theta_0,
                                y=theta_1))

fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor='limegreen',
                                  project_z=True))
fig.show()

# print(linear_cost_matrix)
# y, x = np.meshgrid(theta_1, theta_0)
#
# ax = plt.axes(projection='3d')
# ax.plot_surface(x, y, linear_cost_matrix)
#
# plt.show()



