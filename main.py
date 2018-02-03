'''
https://www.shiyanlou.com/courses/782/labs/2647/document
NBA常规赛结果预测——利用Python进行比赛数据分析
利用scikit-learn提供的Logisitc Regression方法进行回归模型的训练
'''

# -*- coding;utf-8 -*-
import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

#默认elo等级划分
base_elo = 1600
team_elos = {}
team_stats = {}
x = []
y = []
#存放数据目录
folder = 'data'

#初始化函数，从T、O和M表格中读入数据，去除一些无关数据并将这三个表格通过Team属性列进行连接
def initialize_data(Mstat,Ostat,Tstat):
	new_Mstat = Mstat.drop(['RK','Arena'],axis=1)
	new_Ostat = Ostat.drop(['RK','G','MP'],axis=1)
	new_Tstat = Tstat.drop(['RK','G','MP'],axis=1)

	team_stats1 = pd.merge(new_Mstat,new_Ostat,new_Tstat,how='left',on='Team')
	team_stats1 = pd.merge(team_stats1,new_Tstat,how='left',on='Team')
	return team_stats1.set_index('Team',inplace=False,drop=True)

#获取各队的elo score，若无分级，则取默认值base_elo
def get_elo(team):
	try:
		return team_elos[team]
	except:
		team_elos[team] = base_elo
		return team_elos[team]

#计算各队的Elo分级
def calc_elo(win_team,lose_team):
	winner_rank = get_elo(win_team)
	loser_rank = get_elo(lose_team)

	rank_diff = winner_rank - loser_rank
	exp = (rank_diff * -1) / 400
	odds = 1/(1+math.pow(10,exp))

	if winner_rank < 2100:
		k = 32
	elif winner_rank < 2400 and winner_rank >= 2100:
		k = 24
	else:
		k = 16

	new_winner_rank = round(winner_rank + (k * (1 - odds)))
	new_rank_diff = new_winner_rank - winner_rank
	new_loser_rank = loser_rank - new_rank_diff

	return new_winner_rank,new_loser_rank

#建立比赛数据集
def build_dataSet(all_data):
	print('Building data set..')
	x = []
	y = []
	skip = 0
	for index,row in alldata.iterrows():
		Wteam = row['Wteam']
		Lteam = row['Lteam']

		#获取最初两队的elo值
		team1_elo = get_elo(Wteam)
		team2_elo = get_elo(Lteam)

		#主场队伍加100点elo值
		if row['WLoc'] == 'N':
			team1_elo += 100
		else:
			team2_elo += 100

		#将elo当作评价每个队伍的第一个特征值
		team1_features = [team1_elo]
		team2_features = [team2_elo]

		#添加网上获得的各队伍的统计信息
		for key, value in team_stats.loc[Wteam].iteritems():
			team1_features.append(value)
		for key, value in team_stats.loc[Lteam].iteritems():
			team2_features.append(value)

		#将两支队伍的特征值分散在每场比赛数据的左右两侧
		if random.random() > 0.5:
			x.append(team1_features + team2_features)
			y.append(0)
		else:
			x.append(team2_features + team1_features)
			y.append(1)

		if skip = 0:
			print x
			skip = 1

		#根据这场比赛的数据更新队伍的elo值
		new_winner_rank,new_loser_rank = calc_elo(Wteam,Lteam)
		team_elos[Wteam] = new_winner_rank
		team_elos[Lteam] = new_loser_rank
	return np.nan_to_num(X),y

def predict_winner(team_1, team_2, model):
    features = []

    # team 1,客场
    features.append(get_elo(team_1))
    for key, value in team_stats.loc[team_1].iteritems():
        features.append(value)

    # team 2，主场
    features.append(get_elo(team_2) + 100)
    for key, value in team_stats.loc[team_2].iteritems():
        features.append(value)

    features = np.nan_to_num(features)
    return model.predict_proba([features])

if __name__ == '__main__':

	Mstat = pd.read_csv(folder + '/15-16Miscellaneous_Stat.csv')
    Ostat = pd.read_csv(folder + '/15-16Opponent_Per_Game_Stat.csv')
    Tstat = pd.read_csv(folder + '/15-16Team_Per_Game_Stat.csv')

    team_stats = initialize_data(Mstat, Ostat, Tstat)

    result_data = pd.read_csv(folder + '/2015-2016_result.csv')
    X, y = build_dataSet(result_data)

    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))

    model = linear_model.LogisticRegression()
    model.fit(X, y)

    #利用10折交叉验证计算训练正确率
    print("Doing cross-validation..")
    print(cross_val_score(model, X, y, cv = 10, scoring='accuracy', n_jobs=-1).mean())
    

    #利用训练好的model在16-17年的比赛中进行预测
    print('Predicting on new schedule..')
    schedule1617 = pd.read_csv(folder + '/16-17Schedule.csv')
    result = []
    for index, row in schedule1617.iterrows():
        team1 = row['Vteam']
        team2 = row['Hteam']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            result.append([winner, loser, prob])
        else:
            winner = team2
            loser = team1
            result.append([winner, loser, 1 - prob])

    with open('16-17Result.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['win', 'lose', 'probability'])
        writer.writerows(result)