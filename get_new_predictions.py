#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:54:13 2024

@author: ryandrost
"""
#from train import plot_and_df_with_preds
import joblib
import pickle
import numpy as np
import pandas as pd
#from sklearn.linear_model import Lasso

def input_int_data(text, x_min, x_max):
    '''
    Ensures live input is an integer in a specified range

    Parameters
    ----------
    text : string
        text to be displayed for the input function
    x_min : int
        minimum value allowed as input
    x_max : int
        maximum value allowed as input

    Returns
    -------
    int
        input as an integer

    '''
    x = input(text)
    while type(x) == str:
        try:
            int(x)
        except ValueError:
            x = input(f"Invalid entry. Value must be integer between {x_min} and {x_max}: ")
        else:
            if (int(x) < x_min) | (int(x) > x_max):
                x = input(f"Invalid entry. Value must be integer between {x_min} and {x_max}: ")
            else:
                return int(x)
            
model_dict = {'1': 'basic',
              '2': 'basic_with_leg_trophy',
              '3': 'users',
              '4': 'full',
              '5': 'all'}

skip = input("You can use this to see projections for attacks.\n(Press ENTER, or type 'h' for instructions)\n")

if skip == 'h':
    print("There are several different models.\n--Basic: Assumes a generic attacker. Good for predicting an attack with readily available information.")
    print("--Basic with general user ability: Adjusts the basic projection factoring in the general level of the player.")
    print("--Basic with specific user ability: Adjusts the basic projection factoring in the specific player's attack history.")
    input("--Full: Most accurate projection; uses additional information that may not be as available before an attack.\n(Press ENTER)")

m = input("\nChoose the projection you want to see:\n'1': basic\n'2': basic with general user ability\n'3': basic with specific user ability\n'4': full\n'5': all\n").lower()
while m not in model_dict.keys():
    m = input("Invalid entry. Type '1', '2', '3', '4', or '5'\n")

model_name = model_dict[m]
m = int(m)

th_att = input_int_data("Attacking town hall level: ", 1, 16)
th_def = input_int_data("Defending town hall level: ", 1, 16)
ww = input_int_data("Gold in defending town hall: ", 0, 35000)
gs = input_int_data("Number of gold storages (defender): ", 0, 4)
cc_def = input_int_data("Troops in defending clan castle: ", 0, 50)
army_size = input_int_data("Army size (from army camps): ", 0, 320)
cc_att = input_int_data("Troops in attacking clan castle: ", 0, 50)
bk = input_int_data("BK level (0 if not available): ", 0, 95)
aq = input_int_data("AQ level (0 if not available): ", 0, 95)
gw = input_int_data("GW level (0 if not available): ", 0, 70)
rc = input_int_data("RC level (0 if not available): ", 0, 45)
first = input("First attack on this base in the war? (Y or N): ").lower()
while first not in ['y', 'n']:
    first = input("Invalid entry. Value must be 'Y' or 'N': ").lower()

hero_sum = bk + aq + gw + rc
base_ww_defend = ww * (gs+1) / 1000
th_attack_lower = 1 if th_att < th_def else 0
th_attack_higher = 1 if th_att > th_def else 0
first = 0 if first.lower() == 'n' else 1
army_size_total = army_size + cc_att

if m in [2, 4, 5]:
    leg_trophy = input_int_data("Attacker's legend trophies: ", 0, 100000)
    
if m in [3, 4, 5]:
    users = pd.read_csv('data/members.csv')
    user = input("Enter attacker username ('Other' for a generic attacker): ").lower()
    while (user not in list(users['user_name'].str.lower())) & (user != 'other'):
        for u in users['user_name']:
            print(u)
        print('Other')
        user = input("Invalid entry. Enter one of the above options: ").lower()
    user_0 = 1 if user == 'other' else 0
    user_10001 = 1 if user == 'vick' else 0
    user_10002 = 1 if user == 'mctau' else 0
    user_10003 = 1 if user == 'bobo8060' else 0
    user_10004 = 1 if user == 'you' else 0
    user_10005 = 1 if user == 'laney' else 0
    user_10006 = 1 if user == 'slob' else 0
    user_10007 = 1 if user == 'matare' else 0
    user_10008 = 1 if user == 'fatmess' else 0
    user_10009 = 1 if user == 'rytwin' else 0
    user_10010 = 1 if user == 'mitwin' else 0
    user_10011 = 1 if user == 'rimo22' else 0
    user_10012 = 1 if user == 'drewbaby226' else 0
    user_10013 = 1 if user == 'drewdaddy' else 0
    user_10014 = 1 if user == 'aileen' else 0
    user_10015 = 1 if user == 'atom' else 0
    user_10016 = 1 if user == 'kriz' else 0
    user_10017 = 1 if user == 'Qelthar' else 0
    user_10018 = 1 if user == 'goezonme' else 0

if m in [4, 5]:
    bk_eq1 = input_int_data("BK equipment level #1: ", 0, 27)
    bk_eq2 = input_int_data("BK equipment level #2: ", 0, 27)
    bk_pet = input_int_data("BK pet level: ", 0, 15)
    aq_eq1 = input_int_data("AQ equipment level #1: ", 0, 27)
    aq_eq2 = input_int_data("AQ equipment level #2: ", 0, 27)
    aq_pet = input_int_data("AQ pet level: ", 0, 15)
    gw_eq1 = input_int_data("GW equipment level #1: ", 0, 27)
    gw_eq2 = input_int_data("GW equipment level #2: ", 0, 27)
    gw_pet = input_int_data("GW pet level: ", 0, 15)
    rc_eq1 = input_int_data("RC equipment level #1: ", 0, 27)
    rc_eq2 = input_int_data("RC equipment level #2: ", 0, 27)
    rc_pet = input_int_data("RC pet level: ", 0, 15)
    hero_strength = hero_sum + bk_eq1 + bk_eq2 + bk_pet + aq_eq1 + aq_eq2 + aq_pet + \
        gw_eq1 + gw_eq2 + gw_pet + rc_eq1 + rc_eq2 + rc_pet

    bb = input("Attack using battle blimp siege machine? (Y or N): ").lower()
    while bb not in ['y', 'n']:
        bb = input("Attack using battle blimp siege machine? (Y or N): ").lower()
    invis = input_int_data("Number of invisibility spells: ", 0, 14)
    cc_suparch = input_int_data("Number of super archers in attacking clan castle: ", 0, 4)
    cc_supwiz = input_int_data("Number of super wizards in attacking clan castle: ", 0, 5)
    if (10*cc_supwiz + 12*cc_suparch > 0.5) & (invis > 0) & (bb == 1):
        blizz = 1
    else:
        blizz = 0


def scale_columns(df, model_name):
    '''
    scale variable columns to be used for predictions

    Parameters
    ----------
    df : dataframe
        dataframe with new observation(s)
    model_name : string
        name of model used in the file name (to look up proper scaling factors)

    Returns
    -------
    df : dataframe
        dataframe with scaled observation(s)

    '''
    with open(f'models/logistic/3_stars/{model_name}/scaling_std_dict.pkl', 'rb') as f:
        std_dict = pickle.load(f)    
    with open(f'models/logistic/3_stars/{model_name}/scaling_mean_dict.pkl', 'rb') as f:
        mean_dict = pickle.load(f)
    
    for c in df.columns.values:
        if c in std_dict.keys():
            df[c] = (df[c] - mean_dict[c]) / std_dict[c]
    
    return df


if (m == 1) | (m == 5):
    X1 = np.array([[hero_sum, cc_def, base_ww_defend, th_attack_lower,
                    th_attack_higher, first, army_size_total]])
    model_name = 'basic' if m == 5 else model_name
    col_names1 = pd.read_csv(f'models/logistic/3_stars/{model_name}/predictions.csv').columns[1:X1.shape[1]+1]
    df1 = pd.DataFrame(X1)
    df1.columns = col_names1
    df1 = scale_columns(df1, model_name)
    
    model1 = joblib.load(f'models/logistic/3_stars/{model_name}/model.pkl')
    y_pred1 = model1.predict(np.array(df1).reshape(-1,len(df1.columns)))[0]
    y_prob1 = round(model1.predict_proba(np.array(df1).reshape(-1,len(df1.columns)))[:, 1][0], 3)
    
    pred1 = '3 stars' if y_pred1 == 1 else 'Not 3 stars'
    print(f'\n{model_name}\nPrediction: {pred1} \nProbability: {y_prob1}')
    
if (m == 2) | (m == 5):
    X2 = np.array([[hero_sum, cc_def, base_ww_defend, th_attack_lower,
                    th_attack_higher, first, army_size_total, leg_trophy]])
    model_name = 'basic_with_leg_trophy' if m == 5 else model_name
    col_names2 = pd.read_csv(f'models/logistic/3_stars/{model_name}/predictions.csv').columns[1:X2.shape[1]+1]
    df2 = pd.DataFrame(X2)
    df2.columns = col_names2
    df2 = scale_columns(df2, model_name)
    
    model2 = joblib.load(f'models/logistic/3_stars/{model_name}/model.pkl')
    y_pred2 = model2.predict(np.array(df2).reshape(-1,len(df2.columns)))[0]
    y_prob2 = round(model2.predict_proba(np.array(df2).reshape(-1,len(df2.columns)))[:, 1][0], 3)
    
    pred2 = '3 stars' if y_pred2 == 1 else 'Not 3 stars'
    print(f'\n{model_name}\nPrediction: {pred2} \nProbability: {y_prob2}')
    
if (m == 3) | (m == 5):
    X3 = np.array([[hero_sum, cc_def, base_ww_defend, th_attack_lower,
                    th_attack_higher, first, army_size_total,
                    user_0, user_10001, user_10002, user_10003, user_10004, user_10005,
                    user_10006, user_10007, user_10008, user_10010, user_10011,
                    user_10014, user_10015]])
    model_name = 'users' if m == 5 else model_name
    col_names3 = pd.read_csv(f'models/logistic/3_stars/{model_name}/predictions.csv').columns[1:X3.shape[1]+1]
    df3 = pd.DataFrame(X3)
    df3.columns = col_names3
    df3 = scale_columns(df3, model_name)
    
    model3 = joblib.load(f'models/logistic/3_stars/{model_name}/model.pkl')
    y_pred3 = model3.predict(np.array(df3).reshape(-1,len(df3.columns)))[0]
    y_prob3 = round(model3.predict_proba(np.array(df3).reshape(-1,len(df3.columns)))[:, 1][0], 3)
    
    pred3 = '3 stars' if y_pred3 == 1 else 'Not 3 stars'
    print(f'\n{model_name}\nPrediction: {pred3} \nProbability: {y_prob3}')
    
if (m == 4) | (m == 5):
    X4 = np.array([[hero_strength, cc_def, base_ww_defend, th_attack_lower,
                    th_attack_higher, first, army_size_total,
                    user_0, user_10001, user_10002, user_10003, user_10004, user_10005,
                    user_10006, user_10007, user_10008, user_10010, user_10011,
                    user_10014, user_10015, leg_trophy, blizz]])
    model_name = 'full' if m == 5 else model_name
    col_names4 = pd.read_csv(f'models/logistic/3_stars/{model_name}/predictions.csv').columns[1:X4.shape[1]+1]
    df4 = pd.DataFrame(X4)
    df4.columns = col_names4
    df4 = scale_columns(df4, model_name)
    
    model4 = joblib.load(f'models/logistic/3_stars/{model_name}/model.pkl')
    y_pred4 = model4.predict(np.array(df4).reshape(-1,len(df4.columns)))[0]
    y_prob4 = round(model4.predict_proba(np.array(df4).reshape(-1,len(df4.columns)))[:, 1][0], 3)
    
    pred4 = '3 stars' if y_pred4 == 1 else 'Not 3 stars'
    print(f'\n{model_name}\nPrediction: {pred4} \nProbability: {y_prob4}')


