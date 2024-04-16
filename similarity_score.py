#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 01:40:33 2024

@author: ryandrost
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance

df = pd.read_csv('data/attack_data_for_model.csv')

df_filter = df[df['user_id_attack'] == 10001]

i = df.columns.get_loc('barb_pct')
j = df.columns.get_loc('suphog_pct')
df2 = df_filter.iloc[:, i:j].reset_index(drop=True)
X = df2.iloc[0, :]
Y = df2.iloc[9, :]
dist = distance.euclidean(X, Y)
