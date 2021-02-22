
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import scipy.optimize as optimize
#data reading and cleaning
import gc
from tqdm import tqdm
print('Preprocessing may take about 5 minutes')
items = pd.read_csv('items.csv')
samples = pd.read_csv('samples.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
users = pd.read_csv('users.csv')


train_cols ={
    'User ID':'usr_id',
    'Item1 ID':'item1_id',
    'Item2 ID':'item2_id',
    'Preference':'preference'
}
train = train.rename(columns=train_cols)

test_cols ={
    'User ID':'usr_id',
    'Item1 ID':'item1_id',
    'Item2 ID':'item2_id',
}
test = test.rename(columns=test_cols)

users_cols ={
    'User ID':'usr_id',
    ' Education':'education',
    ' Age':'age',
    ' Gender':'gender',
    ' Region':'region'
}
users = users.rename(columns=users_cols)

# data merging

item1_cols ={
    'Item ID':'item1_id',
    ' BodyType':'body_type_1',
    ' Transmission':'transmission_1',
    ' Engin Capacity':'engin_capacity_1',
    ' Fuel Consumed':'fuel_consumed_1'
}

item2_cols ={
    'Item ID':'item2_id',
    ' BodyType':'body_type_2',
    ' Transmission':'transmission_2',
    ' Engin Capacity':'engin_capacity_2',
    ' Fuel Consumed':'fuel_consumed_2'
}

def find_winner():
    k = pd.read_csv('train.csv')
    train_cols ={
        'User ID':'usr_id',
        'Item1 ID':'item1_id',
        'Item2 ID':'item2_id',
        'Preference':'preference'
    }
    k = k.rename(columns=train_cols)

    win=list()
    for index in k.index:
        if k.preference[index] ==0:
            win.append(k.item1_id[index])
        else:
            win.append(k.item2_id[index])
    k['win'] = win
    
    return k

def number_count(k):
    k = k.drop(columns=['preference','win'])

    p1 = k.pivot_table(index = ['usr_id'],columns=['item1_id'], aggfunc='count').fillna(0)
    p2= k.pivot_table(index = ['usr_id'],columns=['item2_id'], aggfunc='count').fillna(0)

    p1.columns = p1.columns.to_series().str.join('_')
    p2.columns = p2.columns.to_series().str.join('_')

    p1.columns = ['1_count','2_count','3_count','4_count','5_count',
                 '6_count','7_count','8_count','9_count','10_count']

    p2.columns = ['1_count','2_count','3_count','4_count','5_count',
                 '6_count','7_count','8_count','9_count','10_count']
    del p1.index.name, p2.index.name
    return p1+p2

def winner_count(k):
    k = k.drop(columns=['item1_id','item2_id'])

    result = k.pivot_table(index = ['usr_id'],columns=['win'], aggfunc='count')

    result.columns.to_series().str.join('_')
    del result.index.name

    #result.columns = result.columns.get_level_values(0)
    result.columns = ['1_count','2_count','3_count','4_count','5_count',
                 '6_count','7_count','8_count','9_count','10_count']
    return result

def occur_rate(k):
    k = (k/9).fillna(0)
    k['usr_id'] = k.index
    
    return k

def win_rate(w_count,n_count):
    k = (w_count/n_count).fillna(0)
    k.columns = ['1_win','2_win','3_win','4_win','5_win',
                '6_win','7_win','8_win','9_win','10_win']
    k['usr_id'] = k.index
    
    return k
    
def max_f(params):
    r1,r2,r3,r4,r5,r6,r7,r8,r9,r10 = params
    R = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10]
    index:int
    result = 1
    for index in data.index:
        result *= R[data.win[index]-1] / (R[data.item1_id[index]-1] + R[data.item2_id[index]-1])
        
    return -result


#calculate occur times and win rate

winner_list = find_winner()
n_count = number_count(winner_list)
w_count = winner_count(winner_list)
occur = occur_rate(n_count)
win = win_rate(w_count,n_count)

#concat train and test
df = pd.concat([train,test])
item_1 = items.copy().rename(columns = item1_cols)
item_2 = items.copy().rename(columns = item2_cols)
df = pd.merge(df,users, on =['usr_id'])
df = pd.merge(df,item_1, on =['item1_id'])
df = pd.merge(df,item_2, on =['item2_id'])
df.shape

del item_1,item_2
gc.collect()
#feature generation
#df = pd.merge(df,occur,on = ['usr_id'])
df = pd.merge(df,win,on = ['usr_id'])
df['engin2_bigger'] = 0
df.engin2_bigger[df.engin_capacity_1==df.engin_capacity_2] = 2
df.engin2_bigger[df.engin_capacity_1>df.engin_capacity_2] = 0
df.engin2_bigger[df.engin_capacity_1<df.engin_capacity_2] = 1

del n_count,w_count,occur,win
gc.collect()

#calculate likelyhood

r_cobyla_list = list()
r_powell_list = list()
initial_guess = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
for i in tqdm(range(60)):
    data = winner_list[winner_list['usr_id']==i]
    result_cobyla = optimize.minimize(max_f, initial_guess, method='COBYLA')
    result_powell = optimize.minimize(max_f, initial_guess, method='Powell')
    r_cobyla_list.append(result_cobyla.x - result_cobyla.x.min() + 1)
    r_powell_list.append(result_powell.x - result_powell.x.min()+1)
    

powell = list()
cobylab = list()
for index in tqdm(df.index):
    R_powell = r_powell_list[df.usr_id[index]-1]
    R_cobylab = r_cobyla_list[df.usr_id[index]-1]
    prob_p = R_powell[df.item2_id[index]-1] / (R_powell[df.item1_id[index]-1] + R_powell[df.item2_id[index]-1])
    prob_c = R_powell[df.item2_id[index]-1] / (R_powell[df.item1_id[index]-1] + R_powell[df.item2_id[index]-1])
    
    powell.append(prob_p)
    cobylab.append(prob_c)
    

df['powell'] = powell
#df['cobylab'] = cobylab

del powell#, cobylab
gc.collect()
#split train test
test = df[df.preference.isna()]
train = df[df.preference.isna()==0]

label = train.preference

train = train.drop(columns=['preference'])
test = test.drop(columns=['preference'])

#train = train.drop(columns=['preference','usr_id','item1_id','item2_id'])
#test = test.drop(columns=['preference','usr_id','item1_id','item2_id'])
#train_test split

from sklearn.model_selection import train_test_split
train_data, valid_data, train_target, valid_target = train_test_split(train, label, test_size=0.33, random_state=9)




# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
classifier = LogisticRegression(C= 5.671226331427315, penalty= 'l1')  # 使用类，参数全是默认的  
classifier.fit(train_data, train_target)  # 训练数据来学习，不需要返回值  
#print(accuracy_score(valid_target, classifier.predict(valid_data)))

lv2_logit = classifier.predict_proba(valid_data)[:,1]
lv2_logit_test = classifier.predict_proba(test)[:,1]
score = accuracy_score(valid_target,classifier.predict(valid_data))
print('Logistic Regression Accuracy:{}'.format(score))

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(leaf_size= 29, n_neighbors= 9, p= 1, weights= 'distance')
neigh.fit(train_data,train_target) 

lv2_knn = neigh.predict_proba(valid_data)[:,1]
lv2_knn_test = neigh.predict_proba(test)[:,1]
score = accuracy_score(valid_target, neigh.predict(valid_data))
print('KNN Accuracy:{}'.format(score))

import lightgbm as lgb

#test = test.drop(columns=['User-Item1-Item2','Preference'])
clf = lgb.LGBMClassifier(bagging_fraction= 0.6508296012331772, bagging_freq= 1, boost= 'gbdt',
                         feature_fraction= 0.3474516582970086, learning_rate= 0.16475833466929507,
                         metric= 'binary_logloss', min_data_in_leaf= 52, num_leaves= 40, num_threads= 2,
                         objective= 'binary', tree_learner= 'data')

clf.fit(train_data, train_target)

lv2_lgb = clf.predict_proba(valid_data)[:,1]
lv2_lgb_test = clf.predict_proba(test)[:,1]
score = accuracy_score(valid_target, clf.predict(valid_data))
print('LGBM Accuracy:{}'.format(score))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
NB = gnb.fit(train_data, train_target)

lv2_nb = NB.predict_proba(valid_data)[:,1]
lv2_nb_test = NB.predict_proba(test)[:,1]
score = accuracy_score(valid_target, NB.predict(valid_data))
print('Naive Bayes Accuracy:{}'.format(score))

from sklearn.svm import SVC
svc = SVC(C= 4.460710230756742, gamma= 0.013244596886327797, kernel= 'poly',probability=True)
svm = svc.fit(train_data, train_target)

lv2_svm = svm.predict_proba(valid_data)[:,1]
lv2_svm_test = svm.predict_proba(test)[:,1]
score = accuracy_score(valid_target, svm.predict(valid_data))
print('SVM Accuracy:{}'.format(score))

'''# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# create model

model = Sequential()
model.add(Dense(128, input_dim=27, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
#model.add(Dropout(p=0.01))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_data, train_target, epochs=10, batch_size=10)
# evaluate the model
scores = model.evaluate(valid_data, valid_target)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

lv2_keras = model.predict(valid_data)
lv2_keras_test = model.predict(test)'''

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation= 'tanh', alpha= 1.6005926425406648e-08, hidden_layer_sizes= 83, solver= 'adam',
                   random_state=42)
mlp.fit(train_data, train_target)

lv2_nn = mlp.predict_proba(valid_data)[:,1]
lv2_nn_test = mlp.predict_proba(test)[:,1]
score = accuracy_score(valid_target, mlp.predict(valid_data))
print('Neural Network Accuracy:{}'.format(score))

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(bootstrap= 0, criterion= 'entropy', max_depth=24,
                                    max_features= 0.3090928393734417, min_samples_leaf= 2,
                                    n_estimators= 30, random_state= 40)
classifier.fit(train_data, train_target)  

lv2_RF = classifier.predict_proba(valid_data)[:,1]
lv2_RF_test = classifier.predict_proba(test)[:,1]
score = accuracy_score(valid_target, classifier.predict(valid_data))
print('Random Forest Accuracy:{}'.format(score))

lv2_df = np.column_stack((lv2_logit,lv2_knn,lv2_lgb,lv2_nb,lv2_nn,lv2_svm,lv2_RF))
lv2_df = pd.DataFrame(lv2_df,columns=['logit', 'knn', 'lgb', 'nb','nn','svm','rf'])

qtarget = valid_target.reset_index()['preference']
#lv2_df['preference'] = qtarget
lv2_train_data, lv2_valid_data, lv2_train_target, lv2_valid_target = train_test_split(lv2_df, qtarget, test_size=0.33, random_state=9)

lv2_df_test = np.column_stack((lv2_logit_test,lv2_knn_test,lv2_lgb_test,lv2_nb_test,lv2_nn_test
                               ,lv2_svm_test,lv2_RF_test))
lv2_df_test = pd.DataFrame(lv2_df_test,columns=['logit', 'knn', 'lgb', 'nb','nn','svm','rf'])


# In[ ]:


clf = lgb.LGBMClassifier(bagging_fraction= 0.6409803582029194, bagging_freq= 2, boost= 'dart',
                         feature_fraction= 0.7076381520129653, learning_rate= 0.09672900472668634,
                         min_data_in_leaf= 27, num_leaves= 74, tree_learner= 'serial')
clf.fit(train,label)
# 预测测试集
prediction = clf.predict(test)

k = test.copy()
k["User-Item1-Item2"] = test['usr_id'].map(str)+'-'+test['item1_id'].map(str)+'-'+test['item2_id'].map(str)
k['Preference'] = prediction.astype(np.int8)

submission = k[['User-Item1-Item2','Preference']]

submission.to_csv('Solo_Lgb.csv',index = False)

mlp = MLPClassifier(activation= 'tanh', alpha= 3.3527294273171613e-09, hidden_layer_sizes= 7, solver= 'sgd',
                    random_state=42)
mlp.fit(lv2_df, qtarget)

prediction = mlp.predict(lv2_df_test)

k = test.copy()
k["User-Item1-Item2"] = test['usr_id'].map(str)+'-'+test['item1_id'].map(str)+'-'+test['item2_id'].map(str)
k['Preference'] = prediction.astype(np.int8)

submission = k[['User-Item1-Item2','Preference']]

submission.to_csv('Ensemble_mlp.csv',index = False)

