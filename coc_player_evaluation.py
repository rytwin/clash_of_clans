#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:24:15 2024

@author: ryandrost
"""

import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('models/logistic/3_stars/users/predictions.csv')
users = pd.read_csv('data/members.csv')
data = pd.read_csv('data/attack_data_for_model.csv')

data.rename(columns = {'user_id_attack': 'user_id'}, inplace=True)
data = data[['attack_id', 'user_id']].merge(users, how='left', on='user_id').fillna('opponent')
df = data.merge(df, how='right', on='attack_id')

with open('models/logistic/3_stars/users/scaling_std_dict.pkl', 'rb') as f:
    scaling_std = pickle.load(f)
    
with open('models/logistic/3_stars/users/scaling_mean_dict.pkl', 'rb') as f:
    scaling_mean = pickle.load(f)

for c in df.columns.values:
    if c in scaling_std.keys():
        df[c] = df[c] * scaling_std[c] + scaling_mean[c]
        
df_eval = df.groupby('user_name').agg({'pred_prob_error': ['sum', 'mean', 'count']})
col_index = df_eval.columns.droplevel(0)
df_eval.columns = col_index

new_cols = []
for c in df_eval.columns.values:
    new_cols.append(c + '_pred_error')

df_eval.columns = new_cols
