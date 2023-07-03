import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

data = pd.read_csv("USA_Housing.csv")

bedroom_weight = data['Avg. Area Number of Bedrooms'].to_numpy()
age_weight = data['Avg. Area House Age'].to_numpy()
income_weight = data['Avg. Area Income'].to_numpy()
rooms_weight = data['Avg. Area Number of Rooms'].to_numpy()
population_weight = data['Area Population'].to_numpy()
price_weight = data['Price'].to_numpy()

scaled_bedroom = (bedroom_weight - np.mean(bedroom_weight)) / (np.max(bedroom_weight) - np.min(bedroom_weight))
scaled_age = (age_weight - np.mean(age_weight)) / (np.max(age_weight) - np.min(age_weight))
scaled_income = (income_weight - np.mean(income_weight)) / (np.max(income_weight) - np.min(income_weight))
scaled_rooms = (rooms_weight - np.mean(rooms_weight)) / (np.max(rooms_weight) - np.min(rooms_weight))
scaled_population = (population_weight - np.mean(population_weight)) / (np.max(population_weight) -
                                                                        np.min(population_weight))

x = data.iloc[:, 0:4]
y = data.iloc[:, -2]

learning_rate = 0.01

# apply SelectKBest class to extract top 5 feature
best_features = SelectKBest(score_func=f_regression, k=4)
fit = best_features.fit(x, y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x.columns)

features_scores = pd.concat([df_columns, df_scores], axis=1)
features_scores.columns = ['specs', 'scores']
print(features_scores)

slopes = np.array([0, 0, 0, 0, 0])
sub_slope = np.array([0, 0, 0, 0, 0])
intercept = 0
sub_intercept = 0

iteration = []
jw = np.array([])


def cal_derived():
    matrix = np.hstack((scaled_income.reshape(-1, 1), scaled_population.reshape(-1, 1), scaled_age.reshape(-1, 1),
                        scaled_rooms.reshape(-1, 1), scaled_bedroom.reshape(-1, 1)))
    sub_y_hat = np.dot(matrix, slopes.reshape(-1, 1))
    y_hat = sub_y_hat.ravel() + intercept
    sub_derived = y_hat - price_weight
    return sub_derived


def gradient_decent():
    for j in range(0, len(slopes)):
        sub_derived = cal_derived()
        if j == 0:
            derived = sub_derived * scaled_income
        elif j == 1:
            derived = sub_derived * scaled_population
        elif j == 2:
            derived = sub_derived * scaled_age
        elif j == 3:
            derived = sub_derived * scaled_rooms
        elif j == 4:
            derived = sub_derived * scaled_bedroom

        added_derived = derived.sum()
        sub_slope[j] = added_derived / len(price_weight)


for i in range(100000):
    gradient_decent()
    cal_der = cal_derived()
    sub_intercept = cal_der.sum() / len(price_weight)

    sub_jw = cal_der * cal_der.T
    jw = np.append(jw, [sub_jw.sum() / (2 * len(price_weight))])
    # print(i)
    iteration.append(i)

    slopes = slopes - (learning_rate * sub_slope)
    intercept = intercept - (learning_rate * sub_intercept)
    print(slopes)
    print('intercept', intercept)

print('Estimated price: ',
      (slopes[0] * scaled_income[2000] + slopes[1] * scaled_population[2000] + slopes[2] * scaled_age[2000]
       + slopes[3] * scaled_rooms[2000] + slopes[4] * scaled_bedroom[2000]) + intercept)

print('price', price_weight[2000])

plt.scatter(iteration, jw, c="blue")
plt.show()
