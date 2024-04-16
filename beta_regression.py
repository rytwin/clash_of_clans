#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:50:34 2024

@author: ryandrost
"""

import numpy as np
import pandas as pd
from statsmodels.othermod.betareg import BetaModel

df = pd.read_csv('data/attack_data_for_model.csv')
df['damage'] = df['damage'] / 100
df['th_diff'] = (df['th_diff'] + 15) / 30

X = np.array(df['th_diff'])
Y = np.array(df['damage'])

X_test = X[0:2]

mod = BetaModel(X, Y)
rslt = mod.fit()
params = rslt.params
print(rslt.summary())

Y_test = mod.predict(params)
