import numpy as np
import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import xgboost as xgb
# from keras.models import Sequential
# from keras.layers import Dense, Activation
from sklearn.neural_network import MLPClassifier

# def get_epl_player(player_data):

def calc_team_strength(match, teams):
	total_strength = {}
	strength = {}
	for i in range(0,len(teams)):
		if teams[i] not in strength:
			total_strength[teams[i]] = 0.0
			strength[teams[i]] = 0.0

	no_of_seasons = match["season"].unique()
	# print no_of_seasons
	for i in no_of_seasons:
		for key,value in strength.items():
				total_strength[key] = (0.3 * total_strength[key]) + (0.7 * strength[key])
				strength[teams[i]] = 0.0
		mtc = match[match['season'] == i]
		for index, m in mtc.iterrows():
			# print m['home_team_api_id']
			if m['home_team_goal'] == m['away_team_goal']:
				# print m['home_team_api_id']
				strength[m['home_team_api_id']] += 0.3
			elif m['home_team_goal'] - m['away_team_goal'] > 3:
				strength[m['home_team_api_id']] += 1
				strength[m['away_team_api_id']] -= 0.1
			elif m['home_team_goal'] - m['away_team_goal'] <= 3 and m['home_team_goal'] - m['away_team_goal'] > 0:
				strength[m['home_team_api_id']] += 0.85
				strength[m['away_team_api_id']] -= 0.05
			elif m['away_team_goal'] - m['home_team_goal'] > 3:
				strength[m['away_team_api_id']] += 1
				strength[m['home_team_api_id']] -= 0.1
			elif m['away_team_goal'] - m['home_team_goal'] <= 3 and m['away_team_goal'] - m['home_team_goal'] > 0:
				strength[m['away_team_api_id']] += 0.9
				strength[m['home_team_api_id']] -= 0.1
			else:
				pass
		for key,value in strength.items():
			strength[key] /= 19

	for key,value in total_strength.items():
			strength[key] /= len(no_of_seasons)
	return total_strength

league = pd.read_csv("League.csv")
matches = pd.read_csv("Match.csv")

liga_id = league.loc[league['name'] == 'Spain LIGA BBVA']['country_id'].values[0]

matches = matches[matches["league_id"] == liga_id]

# matches.to_csv("epl_matches.csv")

player_data = pd.read_csv("Player.csv")

team_data = pd.read_csv("Team.csv")
team_attributes = pd.read_csv("Team_Attributes.csv")

team_data = team_data[team_data['team_api_id'].isin(matches["home_team_api_id"].unique())]
# team_data.to_csv("epl_team.csv")
# print team_data

team_attributes = team_attributes[team_attributes['team_api_id'].isin(matches["home_team_api_id"].unique())]
team_attributes = team_attributes.sort_values("team_api_id")

# team_attributes.to_csv("epl_team_attributes.csv")

match_features = ['season','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal']

matches = matches[match_features]

team_names = matches['home_team_api_id'].unique()
seasons = matches['season'].unique()

dic = {}
dic3 = {}
team_long_name = {}
for index, rows in team_data.iterrows():
	team_long_name[rows['team_api_id']] = rows['team_long_name']

count = 1
for i in range(0,len(team_names)):
	if team_names[i] not in dic:
		dic[team_names[i]] = count
		dic3[count] = team_names[i]
		count += 1

dic2 = {}
count = 1
for i in range(0,len(seasons)):
	if seasons[i] not in dic2:
		dic2[seasons[i]] = count
		count += 1

matches['home_team_api_id'] = matches['home_team_api_id'].replace(dic)
matches['away_team_api_id'] = matches['away_team_api_id'].replace(dic)
matches['season'] = matches['season'].replace(dic2)

team_names = matches['home_team_api_id'].unique()

strength = calc_team_strength(matches, team_names)

a = []

for key,value in strength.items():
	for i, j in dic.items():
		if dic[i] == key:
			a.append((value,team_data[team_data['team_api_id'] == i]['team_long_name'].values[0]))


matches['home_win'] = pd.Series(0, index=matches.index)
matches['away_win'] = pd.Series(0, index=matches.index)
matches['draw'] = pd.Series(0, index=matches.index)
matches['home_team_strength'] = pd.Series(0.0, index=matches.index)
matches['away_team_strength'] = pd.Series(0.0, index=matches.index)

for index,row in matches.iterrows():
	# print strength[row['home_team_api_id']]
	matches.set_value(index, 'home_team_strength', strength[row['home_team_api_id']])
	matches.set_value(index, 'away_team_strength', strength[row['away_team_api_id']])
	if row['home_team_goal'] > row['away_team_goal']:
		matches.set_value(index, 'home_win', 1)
	elif row['away_team_goal'] > row['home_team_goal']:
		matches.set_value(index, 'away_win', 1)
	else:
		matches.set_value(index, 'draw', 1)

# print matches.head()

final_features = ['home_team_api_id', 'away_team_api_id', 'home_team_strength', 'away_team_strength']

classes = ['home_win', 'away_win', 'draw']

train_x = matches[matches['season'] != 8][final_features]
test_x = matches[matches['season'] == 8][final_features]
train_y = matches[matches['season'] != 8][classes]
test_y = matches[matches['season'] == 8][classes]

# print train_x.head(), train_y.head()
# for key,value in strength.items():
# 	print key, strength[key]
# print strength
# a = sorted(a)[::-1]
# i = 1
# for j in a:
# 	print "Position: " + str(i) + j[1]
# 	i += 1


"""logistic regression fit --- decent performance"""
clfs = []
for i in train_y:
	clf = LogisticRegression(C=0.05)
	clf.fit(train_x, train_y[i])
	clfs.append(clf)


"""NB --- poor performance""" 
for i in train_y:
	clf = MultinomialNB()
	clf.fit(train_x, train_y[i])
	clfs.append(clf)


"""SVM --- descent"""
for i in train_y:
	clf = svm.SVC(C=1.0, kernel='rbf', degree=3, probability=False)
	clf.fit(train_x, train_y[i])
	clfs.append(clf)

"""xgboost classifier"""
for i in train_y:
	clf =  xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1)
	clf.fit(train_x, train_y[i])
	clfs.append(clf)

"""MLP classifier"""
for i in train_y:
	clf =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)
	clf.fit(train_x, train_y[i])
	clfs.append(clf)

# diction = {"home_team_api_id" : dic[10281], "away_team_api_id" : dic[9864], "home_team_strength": strength[dic[10281]],"away_team_strength": strength[dic[9864]]}
# p = pd.DataFrame([[12][13]], )
# p.to_csv("test1.csv",index=False)

# test = pd.read_csv("test.csv")
# test['home_team_strength'] = pd.Series(0,index=test.index)
# test['away_team_strength'] = pd.Series(0,index=test.index)

# for index, row in test.iterrows():
# 	test.set_value(index, 'home_team_api_id', dic[row['home_team_api_id']])
# 	test.set_value(index, 'away_team_api_id', dic[row['away_team_api_id']])
# 	test.set_value(index, 'home_team_strength', strength[row['home_team_api_id']])
# 	test.set_value(index, 'away_team_strength', strength[row['away_team_api_id']])

# test_x = pd.DataFrame(diction.items())
# ans = []
arr = []
for i in range(0, len(clfs)):
	# arr.append(clfs[i].predict_proba(test_x))
	arr.append(clfs[i].predict(test_x))
# print arr[0]
# # for i in range(0,10):
# # 	if arr[0][i] == arr[1][i]:
# # 		arr[2][i] = 1
# # 	print arr[0][i], arr[1][i], arr[2][i]

# # arr = []
points = {}

for key,value in dic3.items():
	points[team_long_name[dic3[key]]] = 0

j = 0
for index, rows in test_x.iterrows():
	# arr = []
	# for i in range(0, len(clfs)):
	# 	arr.append(clfs[i].predict(test_x[index]))
	# print arr
	# print arr[0][j]
	if arr[0][j] == arr[1][j]:
		# print "in"
		# print dic3[rows['home_team_api_id']], dic3[rows['away_team_api_id']]
		points[team_long_name[dic3[rows['home_team_api_id']]]] += 1
		points[team_long_name[dic3[rows['away_team_api_id']]]] += 1

	elif arr[0][j] > arr[1][j]:
		points[team_long_name[dic3[rows['home_team_api_id']]]] += 3

	else:
		points[team_long_name[dic3[rows['away_team_api_id']]]] += 3

	j += 1

final_standings = []

for key,value in points.items():
	final_standings.append((points[key],key))

final_standings = sorted(final_standings)[::-1]
position = 1
for i in range(0,len(final_standings)):
	print str(position) + str(final_standings[i][1]) + " " + str(final_standings[i][0])
	position += 1
i = 0
score = 0.0
for index, rows in test_y.iterrows():
	if arr[0][i] == arr[1][i]:
		arr[2][i] = 1
	# print arr[0][i], arr[1][i], arr[2][i]
	# print rows['home_win'], rows['away_win'], rows['draw']
	if arr[0][i] == rows['home_win'] and arr[1][i] == rows['away_win'] and arr[2][i] == rows['draw']:
		score += 1.0
	i += 1

print score/float(i-1)