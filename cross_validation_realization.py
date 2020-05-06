import numpy as np
import pandas
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


def init_accuracy(d_xy):
	for j in range(0, 50):
		d_xy['accuracy'][j] = 0.
		d_xy['num'][j] = 0


def data_update(d_xy, i_train, i_test):
	d_xy['X_train'] = []
	d_xy['y_train'] = []
	d_xy['X_test'] = []
	d_xy['y_test'] = []

	for j in range(0, len(i_train)):
		tr_index = i_train[j]
		d_xy['X_train'].append(d_xy['X'][tr_index])
		d_xy['y_train'].append(d_xy['y'][tr_index])

	for j in range(0, len(i_test)):
		t_index = i_test[j]
		d_xy['X_test'].append(d_xy['X'][t_index])
		d_xy['y_test'].append(d_xy['y'][t_index])


def accuracy_calculation(d_xy, response, num_n):
	should_be = d_xy['y_test']
	good_response = 0

	for j in range(0, len(should_be)):
		if response[j] == should_be[j]:
			good_response += 1
	if good_response == 0:
		correct_share = 0
	else:
		correct_share = good_response / len(should_be)
	# if correct_share > d_xy['accuracy'][num_n - 1]:
	d_xy['accuracy'][num_n - 1] += correct_share
	d_xy['num'][num_n - 1] += 1


def knc_testing(d_xy, num_n):
	neigh = KNeighborsClassifier(n_neighbors=num_n)
	neigh.fit(d_xy['X_train'], d_xy['y_train'])
	knc_response = neigh.predict(d_xy['X_test'])
	accuracy_calculation(d_xy, knc_response, num_n)


# main function
data = pandas.read_csv('wines.csv')

wine_class = data['class']
del (data['class'])
wine_pr = []

for i in range(0, 178):
	wine_pr.append(np.array(data.iloc[i, :]))

Xy_data = {
	'X': wine_pr,
	'y': wine_class,
	'X_train': [],
	'y_train': [],
	'X_test': [],
	'y_test': [],
	'accuracy': np.empty(50),
	'num': np.empty(50)
}

init_accuracy(Xy_data)

kf = KFold(n_splits=5, random_state=42, shuffle=True)

for k in range(1, 51):
	for train_index, test_index in kf.split(Xy_data['X']):
		data_update(Xy_data, train_index, test_index)
		knc_testing(Xy_data, k)

for i in range(0, 50):
	Xy_data['accuracy'][i] = Xy_data['accuracy'][i] / Xy_data['num'][i]
# print('accuracy:')
# print(Xy_data['accuracy'])

# print(Xy_data['accuracy'])
# print(Xy_data['num'])
print('\nk_max =', np.argmax(Xy_data['accuracy']) + 1)
print('accuracy_max =', np.max(Xy_data['accuracy']))

print('\nk_min =', np.argmin(Xy_data['accuracy']) + 1)
print('accuracy_min =', np.min(Xy_data['accuracy']))
