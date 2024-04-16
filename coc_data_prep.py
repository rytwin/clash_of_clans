#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:44:24 2023

@author: ryandrost
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.")

# change to True if you want to save the data file as csv to be used for modeling
save_file = False

def add_sm_num(row):
    '''
    converts string elements so they are in the proper format for extract_unit_info function
    
    Parameters
    ----------
    row : string
        string with elements split by a '-'

    Returns
    -------
    string
        adds '-1' to the string, before the first hyphen

    '''
    parts = row.split('-')
    if len(parts) > 1:
        return f'{parts[0]}-1-{parts[1]}'
    else:
        return row

def extract_unit_info(row, unit_type):
    '''
    extract info from strings with this specific format
    
    Parameters
    ----------
    row : string
        string with elements split by a ',' and then split within those elements by a '-'
    unit_type : string
        string that we are looking for within the row variable

    Returns
    -------
    number: int
        the second sub-element (after the first '-'), which should correspond to amount
    level: int
        the third sub-element (after the second '-'), which should correspond to level
    
    returns none if no elements of the string match the unit_type

    '''
    for unit in row.split(','):
        if unit.startswith(unit_type):
            _, number, level = unit.split('-')
            return int(number), int(level)
    return None, None

def create_equip_columns(df, hero_type):
    '''
    create columns for each equipment, so each row has the level of every equipment (many will be zero)

    Parameters
    ----------
    df : dataframe
        dataframe of observations which will be updated with the new columns
    hero_type : string
        must be one of 4 types (see below)
        in all likelihood, you will be looping over all of these types to add columns for each

    Returns
    -------
    None. adds the columns to the dataframe object passed to the function

    '''
    
    if hero_type not in ['bk', 'aq', 'gw', 'rc']:
        print(f"ERROR: hero_type must be 'bk' 'aq' 'gw' or 'rc'. hero_type was '{hero_type}'")
        return
    
    eq1, eq2 = hero_type + 'equip1', hero_type + 'equip2'
    eq1_lvl, eq2_lvl = eq1 + '_lvl', eq2 + '_lvl'
    
    equip_types = set(df[eq1].unique()).union(set(df[eq2].unique()))
    for equip in equip_types:
        df[f'{equip}_lvl'] = 0

    for idx, row in df.iterrows():
        for equip_col, lvl_col in [(eq1, eq1_lvl), (eq2, eq2_lvl)]:
            equip_type = row[equip_col]
            df.at[idx, f'{equip_type}_lvl'] = row[lvl_col]

def create_pet_columns(df, pet_type):
    '''
    create columns for each pet, so each row has the level of every pet (many will be zero)

    Parameters
    ----------
    df : dataframe
        dataframe of observations which will be updated with the new columns
    pet_type : string
        must be one of 9 types ('lassi', 'eowl', 'myak', 'unicorn', 'frosty', 'diggy', 'pliz', 'phoenix', 'sfox')
        in all likelihood, you will be looping over all of these types to add columns for each

    Returns
    -------
    None. adds the columns to the dataframe object passed to the function

    '''
    
    pet_types = set(df['bkpet'].unique()).union(set(df['aqpet'].unique())).union(set(df['gwpet'].unique())).union(set(df['rcpet'].unique()))
    for pet in pet_types:
        df[f'{pet}_lvl'] = 0
        
    for idx, row in df.iterrows():
        for pet_col, lvl_col in [('bkpet', 'bkpet_lvl'), ('aqpet', 'aqpet_lvl'), ('gwpet', 'gwpet_lvl'), ('rcpet', 'rcpet_lvl')]:
            pet_type = row[pet_col]
            df.at[idx, f'{pet_type}_lvl'] = row[lvl_col]

def create_unit_columns(df, unit_info, unit_type, cc = False):
    '''
    create columns for each unit, so each row has the amount and level of every unit (many will be zero)

    Parameters
    ----------
    df : dataframe
        dataframe of observations which will be updated with the new columns
    unit_info : dataframe
        contains generic info for all units
    unit_type: string
        currently should be one of 'spells', 'troops', and 'sm'
        in all likelihood, you will call the function separately for all 3
    cc: boolean (default: FALSE)
        parsing clan castle troops (TRUE) or regular troops (FALSE) 

    Returns
    -------
    errors: list
        a list of lists. each sub-list has the unit_type (string) and attack_id (int)
            which does not match any of the unit types (i.e. is a typo)
        
    also adds the columns directly to the dataframe object passed to the function
    
    '''
    unit_types = unit_info[unit_info['type'] == unit_type]['unit']
    unit_type_list = list(unit_types)
     
    if cc:
        unit_type = 'cc_' + unit_type
        for units in unit_types:
            df[f'cc_{units}_num'], df[f'cc_{units}_lvl'] = zip(*df[unit_type].apply(lambda x: extract_unit_info(x, units)))
    else:
        for units in unit_types:
            df[f'{units}_num'], df[f'{units}_lvl'] = zip(*df[unit_type].apply(lambda x: extract_unit_info(x, units)))
    
    errors = []
    attack_id = 1
    for row in df[unit_type]:
        for unit in row.split(','):
            unit_id = unit.split('-')[0]
            if unit_id not in unit_type_list:
                errors.append([unit_id, attack_id])
        attack_id += 1
            
    return errors

def create_army_comp_column(df, unit_info, category):
    '''
    create column that calculates percentage of troops with a certain characteristic

    Parameters
    ----------
    df : dataframe
        dataframe of observations which will be updated with new columns
    unit_info : dataframe
        contains info on units that is used to determine which units should be included
    category : string
        characteristic that is being assessed (eg.: 'ground' or 'air'). must be a binary column from unit_info df

    Returns
    -------
    None. adds the column to the dataframe object that was passed to the function

    '''
    df[f'{category}_pct'] = int(unit_info[unit_info['unit'] == 'barb'][category]) * df['barb_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'arch'][category]) * df['arch_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'giant'][category]) * df['giant_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'gob'][category]) * df['gob_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'wb'][category]) * df['wb_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'wiz'][category]) * df['wiz_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'balloon'][category]) * df['balloon_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'healer'][category]) * df['healer_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'drag'][category]) * df['drag_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'pekka'][category]) * df['pekka_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'babydrag'][category]) * df['babydrag_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'miner'][category]) * df['miner_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'edrag'][category]) * df['edrag_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'yeti'][category]) * df['yeti_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'drider'][category]) * df['drider_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'etitan'][category]) * df['etitan_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'root'][category]) * df['root_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'minion'][category]) * df['minion_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'hog'][category]) * df['hog_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'valk'][category]) * df['valk_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'golem'][category]) * df['golem_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'witch'][category]) * df['witch_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'lh'][category]) * df['lh_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'bowl'][category]) * df['bowl_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'ig'][category]) * df['ig_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'hh'][category]) * df['hh_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'aw'][category]) * df['aw_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supbarb'][category]) * df['supbarb_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'suparch'][category]) * df['suparch_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'sneakgob'][category]) * df['sneakgob_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supwb'][category]) * df['supwb_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supgiant'][category]) * df['supgiant_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'rockloon'][category]) * df['rockloon_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supwiz'][category]) * df['supwiz_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supdrag'][category]) * df['supdrag_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'infdrag'][category]) * df['infdrag_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supminion'][category]) * df['supminion_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supvalk'][category]) * df['supvalk_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supwitch'][category]) * df['supwitch_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'ih'][category]) * df['ih_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supbowl'][category]) * df['supbowl_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'supminer'][category]) * df['supminer_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'suphog'][category]) * df['suphog_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'ram'][category]) * df['ram_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'cookie'][category]) * df['cookie_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'fc'][category]) * df['fc_pct_tot'] + \
        int(unit_info[unit_info['unit'] == 'azdrag'][category]) * df['azdrag_pct_tot']
    
    
# read in all necessary csv files
bases = pd.read_csv('data/bases.csv')
wars = pd.read_csv('data/wars.csv')
attacks = pd.read_csv('data/attacks.csv')
unit_info = pd.read_csv('data/unit_info.csv')
users = pd.read_csv('data/members.csv')
attacks['clan_def'] = np.where(attacks['clan_att'] == 'home', 'opponent', 'home')

bases['total_gold'] = bases['gold'] * (bases['num_storages'] + 1)
bases['base_ww'] = round(bases['total_gold'] / 1000, 0)

# merge bases with members using user_name and with war using war_id. keep all entries from bases
# merge with attacks. keep all attacks, adding base info (for attacker and defender) and war info to each observation
df0 = pd.merge(bases, users, how='left', on="user_name")
df1 = pd.merge(df0, wars, how='left', on='war_id')
df2 = pd.merge(df1, attacks, how='right', left_on=['war_id', 'clan', 'rank'], right_on=['war_id', 'clan_att', 'attacker'])
df3 = pd.merge(df2, df0, how='left', left_on=['war_id', 'clan_def', 'defender'], right_on=['war_id', 'clan', 'rank'],
               suffixes=('_attack', '_defend'))
# add binary target variable for 3 stars or not
df3['3_stars'] = np.where(df3['stars'] == 3, 1, 0)
df4 = df3[['attack_id', 'user_id_attack', 'user_id_defend', 'game_version', 'cwl_id', 'damage', 'stars', '3_stars', 'th_attack', 'prof_lvl_attack',
           'war_stars_attack', 'leg_trophy_attack', 'leg_trophy_most_recent_attack', 'army_size_attack', 'base_attack_no', 'th_defend', 'prof_lvl_defend',
           'war_stars_defend', 'leg_trophy_defend', 'base_ww_defend', 'th_loc_x_defend', 'th_loc_y_defend', 'cc_size_defend',
           'bk_lvl', 'aq_lvl', 'gw_lvl', 'rc_lvl', 'bkequip1', 'bkequip1_lvl', 'bkequip2', 'bkequip2_lvl', 'bkpet', 'bkpet_lvl',
           'aqequip1', 'aqequip1_lvl', 'aqequip2', 'aqequip2_lvl', 'aqpet', 'aqpet_lvl', 'gwequip1', 'gwequip1_lvl', 'gwequip2',
           'gwequip2_lvl', 'gwpet', 'gwpet_lvl', 'rcequip1', 'rcequip1_lvl', 'rcequip2', 'rcequip2_lvl', 'rcpet', 'rcpet_lvl',
           'troops', 'spells', 'sm', 'cc_troops', 'cc_spells']]


# expand hero, pet, troop, spell, sm info into separate columns. change all NA values to zero
df = df4.copy()
df['sm'] = df['sm'].apply(add_sm_num)

hero_types = ['bk', 'aq', 'gw', 'rc']
[create_equip_columns(df, hero) for hero in hero_types]

pet_types = ['lassi', 'eowl', 'myak', 'unicorn', 'frosty', 'diggy', 'pliz', 'phoenix', 'sfox']
for pet_type in pet_types:
    create_pet_columns(df, pet_type)

errors_troop = create_unit_columns(df, unit_info, 'troops', False)
errors_spell = create_unit_columns(df, unit_info, 'spells', False)
errors_cc_troop = create_unit_columns(df, unit_info, 'troops', True)
errors_cc_spell = create_unit_columns(df, unit_info, 'spells', True)
errors_sm = create_unit_columns(df, unit_info, 'sm', False)

df.fillna(0, inplace=True)


### add more columns which may be good features for modeling

# binary variables for heros, pets, equip and total values for hero, pet, equip
df['bk'] = np.where(df['bk_lvl'] > 0, 1, 0)
df['aq'] = np.where(df['aq_lvl'] > 0, 1, 0)
df['gw'] = np.where(df['gw_lvl'] > 0, 1, 0)
df['rc'] = np.where(df['rc_lvl'] > 0, 1, 0)
df['hero_num'] = df['bk'] + df['aq'] + df['gw'] + df['rc']
df['bkequip_sum'] = df['bkequip1_lvl'] + df['bkequip2_lvl']
df['aqequip_sum'] = df['aqequip1_lvl'] + df['aqequip2_lvl']
df['gwequip_sum'] = df['gwequip1_lvl'] + df['gwequip2_lvl']
df['rcequip_sum'] = df['rcequip1_lvl'] + df['rcequip2_lvl']
df['equip_sum'] = df['bkequip_sum'] + df['aqequip_sum'] + df['gwequip_sum'] + df['rcequip_sum']
df['pet_sum'] = df['bkpet_lvl'] + df['aqpet_lvl'] + df['gwpet_lvl'] + df['rcpet_lvl']
df['hero_sum'] = df['bk_lvl'] + df['aq_lvl'] + df['gw_lvl'] + df['rc_lvl']
df['hero_pet_strength'] = df['hero_sum'] + df['pet_sum']
df['hero_equip_strength'] = df['hero_sum'] + df['equip_sum']
df['hero_total_strength'] = df['hero_sum'] + df['pet_sum'] + df['equip_sum']
df['bk_pet_strength'] = df['bkpet_lvl'] + df['bk_lvl']
df['bk_equip_strength'] = df['bkequip_sum'] + df['bk_lvl']
df['bk_total_strength'] = df['bk_lvl'] + df['bkpet_lvl'] + df['bkequip_sum']
df['aq_pet_strength'] = df['aqpet_lvl'] + df['aq_lvl']
df['aq_equip_strength'] = df['aqequip_sum'] + df['aq_lvl']
df['aq_total_strength'] = df['aq_lvl'] + df['aqpet_lvl'] + df['aqequip_sum']
df['gw_pet_strength'] = df['gwpet_lvl'] + df['gw_lvl']
df['gw_equip_strength'] = df['gwequip_sum'] + df['gw_lvl']
df['gw_total_strength'] = df['gw_lvl'] + df['gwpet_lvl'] + df['gwequip_sum']
df['rc_pet_strength'] = df['rcpet_lvl'] + df['rc_lvl']
df['rc_equip_strength'] = df['rcequip_sum'] + df['rc_lvl']
df['rc_total_strength'] = df['rc_lvl'] + df['rcpet_lvl'] + df['rcequip_sum']
df['myak'] = np.where(df['myak_lvl'] > 0, 1, 0)
df['sfox'] = np.where(df['sfox_lvl'] > 0, 1, 0)
df['phoenix'] = np.where(df['phoenix_lvl'] > 0, 1, 0)
df['eowl'] = np.where(df['eowl_lvl'] > 0, 1, 0)
df['lassi'] = np.where(df['lassi_lvl'] > 0, 1, 0)
df['diggy'] = np.where(df['diggy_lvl'] > 0, 1, 0)
df['pliz'] = np.where(df['pliz_lvl'] > 0, 1, 0)
df['frosty'] = np.where(df['frosty_lvl'] > 0, 1, 0)
df['unicorn'] = np.where(df['unicorn_lvl'] > 0, 1, 0)
df['pet_num'] = df['myak'] + df['sfox'] + df['phoenix'] + df['eowl'] + df['lassi'] + \
    df['diggy'] + df['pliz'] + df['frosty'] + df['unicorn']
df['gg'] = np.where(df['gg_lvl'] > 0, 1, 0)
df['eqboots'] = np.where(df['eqboots_lvl'] > 0, 1, 0)
df['rvial'] = np.where(df['rvial_lvl'] > 0, 1, 0)
df['barbpup'] = np.where(df['barbpup_lvl'] > 0, 1, 0)
df['vstache'] = np.where(df['vstache_lvl'] > 0, 1, 0)
df['ga'] = np.where(df['ga_lvl'] > 0, 1, 0)
df['ivial'] = np.where(df['ivial_lvl'] > 0, 1, 0)
df['healpup'] = np.where(df['healpup_lvl'] > 0, 1, 0)
df['archpup'] = np.where(df['archpup_lvl'] > 0, 1, 0)
df['fa'] = np.where(df['fa_lvl'] > 0, 1, 0)
df['lgem'] = np.where(df['lgem_lvl'] > 0, 1, 0)
df['etome'] = np.where(df['etome_lvl'] > 0, 1, 0)
df['htome'] = np.where(df['htome_lvl'] > 0, 1, 0)
df['rgem'] = np.where(df['rgem_lvl'] > 0, 1, 0)
df['roygem'] = np.where(df['roygem_lvl'] > 0, 1, 0)
df['sshield'] = np.where(df['sshield_lvl'] > 0, 1, 0)
df['hogpup'] = np.where(df['hogpup_lvl'] > 0, 1, 0)
df['hvial'] = np.where(df['hvial_lvl'] > 0, 1, 0)


# siege machine binary
df['sm'] = df['ww_num'] + df['bb_num'] + df['ss_num'] + df['sb_num'] + df['ll_num'] + df['ff_num'] + df['bd_num']


# spell totals, housing space, and percentages (includes clan castle)
spell_types = unit_info[unit_info['type'] == 'spells']['unit']
for s in spell_types:
    df[f'{s}_tot'] = df[f'{s}_num'] + df[f'cc_{s}_num']
    df[f'{s}_space'] = int(unit_info[unit_info['unit'] == s]['housing'])*df[f'{s}_tot']

df['num_spells'] = df['light_space'] + df['bof_space'] + df['heal_space'] + df['rage_space'] + \
    df['jump_space'] + df['freeze_space'] + df['clone_space'] + df['invis_space'] + \
    df['recall_space'] + df['poison_space'] + df['eq_space'] + df['haste_space'] + \
    df['skel_space'] + df['bat_space'] + df['og_space']
print('num_spells > 14:\n', df[df['num_spells'] > 14][['attack_id', 'num_spells']])

for s in spell_types:
    df[f'{s}_pct'] = df[f'{s}_space'] / df['num_spells']


# troop totals, housing space, and percentages
troop_types = unit_info[unit_info['type'] == 'troops']['unit']
for t in troop_types:
    df[f'{t}_space'] = int(unit_info[unit_info['unit'] == t]['housing']) * df[f'{t}_num']

df['army_size'] = df['barb_space'] + df['ram_space'] + df['cookie_space'] + df['arch_space'] + \
    df['giant_space'] + df['gob_space'] + df['wb_space'] + df['wiz_space'] + df['balloon_space'] + \
    df['healer_space'] + df['drag_space'] + df['pekka_space'] + df['babydrag_space'] + \
    df['miner_space'] + df['edrag_space'] + df['yeti_space'] + df['drider_space'] + \
    df['etitan_space'] + df['root_space'] + df['minion_space'] + df['hog_space'] + \
    df['valk_space'] + df['golem_space'] + df['witch_space'] + df['lh_space'] + \
    df['bowl_space'] + df['ig_space'] + df['hh_space'] + df['aw_space'] + \
    df['supbarb_space'] + df['suparch_space'] + df['sneakgob_space'] + df['supwb_space'] + \
    df['supgiant_space'] + df['rockloon_space'] + df['supwiz_space'] + df['supdrag_space'] + \
    df['infdrag_space'] + df['supminion_space'] + df['supvalk_space'] + df['supwitch_space'] + \
    df['ih_space'] + df['supbowl_space'] + df['supminer_space'] + df['suphog_space'] + \
    df['fc_space'] + df['azdrag_space']

for t in troop_types:
    df[f'{t}_pct'] = df[f'{t}_space'] / df['army_size']
    
    
# cc troop totals, housing space, and percentages
for t in troop_types:
    df[f'cc_{t}_space'] = int(unit_info[unit_info['unit'] == t]['housing'])*df[f'cc_{t}_num']

df['cc_num_troops'] = df['cc_barb_space'] + df['cc_arch_space'] + \
    df['cc_giant_space'] + df['cc_gob_space'] + df['cc_wb_space'] + df['cc_wiz_space'] + df['cc_balloon_space'] + \
    df['cc_healer_space'] + df['cc_drag_space'] + df['cc_pekka_space'] + df['cc_babydrag_space'] + \
    df['cc_miner_space'] + df['cc_edrag_space'] + df['cc_yeti_space'] + df['cc_drider_space'] + \
    df['cc_etitan_space'] + df['cc_root_space'] + df['cc_minion_space'] + df['cc_hog_space'] + \
    df['cc_valk_space'] + df['cc_golem_space'] + df['cc_witch_space'] + df['cc_lh_space'] + \
    df['cc_bowl_space'] + df['cc_ig_space'] + df['cc_hh_space'] + df['cc_aw_space'] + \
    df['cc_supbarb_space'] + df['cc_suparch_space'] + df['cc_sneakgob_space'] + df['cc_supwb_space'] + \
    df['cc_supgiant_space'] + df['cc_rockloon_space'] + df['cc_supwiz_space'] + df['cc_supdrag_space'] + \
    df['cc_infdrag_space'] + df['cc_supminion_space'] + df['cc_supvalk_space'] + df['cc_supwitch_space'] + \
    df['cc_ih_space'] + df['cc_supbowl_space'] + df['cc_supminer_space'] + df['cc_suphog_space']

for t in troop_types:
    df[f'cc_{t}_pct'] = df[f'cc_{t}_space'] / df['cc_num_troops']

df.fillna(0, inplace=True)


# total army size (regular and cc troops)
df['army_size_total'] = df['army_size'] + df['cc_num_troops']

for t in troop_types:
    df[f'{t}_pct_tot'] = (df[f'{t}_space'] + df[f'cc_{t}_space']) / df['army_size_total']


# variables for general army composition
create_army_comp_column(df, unit_info, 'air')
create_army_comp_column(df, unit_info, 'ground')
create_army_comp_column(df, unit_info, 'supertroop')
create_army_comp_column(df, unit_info, 'walls')
create_army_comp_column(df, unit_info, 'splash_binary')
create_army_comp_column(df, unit_info, 'range_binary')
create_army_comp_column(df, unit_info, 'support')
df['air_ratio'] = df['air_pct'] / df['ground_pct']
df['air_ratio'].replace([np.inf], 370, inplace=True)


# variables for army strength, measured by distance from max level army
sm_types = unit_info[unit_info['type'] == 'sm']['unit']
for sm in sm_types:
    df[f'{sm}_from_max'] = (df[f'{sm}_lvl'] - int(unit_info[unit_info['unit'] == sm]['max_lvl'])) * df[f'{sm}_num']
    if (df[f'{sm}_from_max'] > 0).any():
        print(df[df[f'{sm}_from_max'] > 0][['attack_id', f'{sm}_from_max']])

df['sm_below_max'] = df['ww_from_max'] + df['bb_from_max'] + df['ss_from_max'] + df['sb_from_max'] + \
    df['ll_from_max'] + df['ff_from_max'] + df['bd_from_max']

for s in spell_types:
    df[f'{s}_from_max_lvl'] = df[f'{s}_lvl'] - int(unit_info[unit_info['unit'] == s]['max_lvl'])
    df[f'{s}_from_max'] = df[f'{s}_pct'] * df[f'{s}_from_max_lvl']
    df[f'{s}_max_pct'] = np.where(df[f'{s}_from_max_lvl'] == 0, df[f'{s}_pct'], 0)
    if (df[f'{s}_from_max_lvl'] > 0).any():
        print(df[df[f'{s}_from_max_lvl'] > 0][['attack_id', f'{s}_from_max_lvl']])

df['spell_avg_below_max'] = df['light_from_max'] + df['bof_from_max'] + df['heal_from_max'] + df['rage_from_max'] + \
    df['jump_from_max'] + df['freeze_from_max'] + df['clone_from_max'] + df['invis_from_max'] + \
    df['recall_from_max'] + df['poison_from_max'] + df['eq_from_max'] + df['haste_from_max'] + \
    df['skel_from_max'] + df['bat_from_max'] + df['og_from_max']
    
df['spell_max_percent'] = df['light_max_pct'] + df['bof_max_pct'] + df['heal_max_pct'] + df['rage_max_pct'] + \
    df['jump_max_pct'] + df['freeze_max_pct'] + df['clone_max_pct'] + df['invis_max_pct'] + \
    df['recall_max_pct'] + df['poison_max_pct'] + df['eq_max_pct'] + df['haste_max_pct'] + \
    df['skel_max_pct'] + df['bat_max_pct'] + df['og_max_pct']

for t in troop_types:
    df[f'{t}_from_max_lvl'] = df[f'{t}_lvl'] - int(unit_info[unit_info['unit'] == t]['max_lvl'])
    df[f'{t}_from_max'] = df[f'{t}_pct'] * df[f'{t}_from_max_lvl']
    df[f'{t}_max_pct'] = np.where(df[f'{t}_from_max_lvl'] == 0, df[f'{t}_pct'], 0)
    df[f'cc_{t}_from_max_lvl'] = df[f'cc_{t}_lvl'] - int(unit_info[unit_info['unit'] == t]['max_lvl'])
    df[f'cc_{t}_from_max'] = df[f'cc_{t}_pct'] * df[f'cc_{t}_from_max_lvl']
    df[f'cc_{t}_max_pct'] = np.where(df[f'cc_{t}_from_max_lvl'] == 0, df[f'cc_{t}_pct'], 0)
    if (df[f'{t}_from_max_lvl'] > 0).any():
        print(df[df[f'{t}_from_max_lvl'] > 0][['attack_id', f'{t}_from_max_lvl']])
    if (df[f'cc_{t}_from_max_lvl'] > 0).any():
        print(df[df[f'cc_{t}_from_max_lvl'] > 0][['attack_id', f'cc_{t}_from_max_lvl']])

df['army_avg_below_max'] = df['barb_from_max'] + df['ram_from_max'] + df['cookie_from_max'] + df['arch_from_max'] + \
    df['giant_from_max'] + df['gob_from_max'] + df['wb_from_max'] + df['wiz_from_max'] + df['balloon_from_max'] + \
    df['healer_from_max'] + df['drag_from_max'] + df['pekka_from_max'] + df['babydrag_from_max'] + \
    df['miner_from_max'] + df['edrag_from_max'] + df['yeti_from_max'] + df['drider_from_max'] + \
    df['etitan_from_max'] + df['root_from_max'] + df['minion_from_max'] + df['hog_from_max'] + \
    df['valk_from_max'] + df['golem_from_max'] + df['witch_from_max'] + df['lh_from_max'] + \
    df['bowl_from_max'] + df['ig_from_max'] + df['hh_from_max'] + df['aw_from_max'] + \
    df['supbarb_from_max'] + df['suparch_from_max'] + df['sneakgob_from_max'] + df['supwb_from_max'] + \
    df['supgiant_from_max'] + df['rockloon_from_max'] + df['supwiz_from_max'] + df['supdrag_from_max'] + \
    df['infdrag_from_max'] + df['supminion_from_max'] + df['supvalk_from_max'] + df['supwitch_from_max'] + \
    df['ih_from_max'] + df['supbowl_from_max'] + df['supminer_from_max'] + df['suphog_from_max'] + \
    df['fc_from_max'] + df['azdrag_from_max']

df['army_max_percent'] = df['barb_max_pct'] + df['ram_max_pct'] + df['cookie_max_pct'] + df['arch_max_pct'] + \
    df['giant_max_pct'] + df['gob_max_pct'] + df['wb_max_pct'] + df['wiz_max_pct'] + df['balloon_max_pct'] + \
    df['healer_max_pct'] + df['drag_max_pct'] + df['pekka_max_pct'] + df['babydrag_max_pct'] + \
    df['miner_max_pct'] + df['edrag_max_pct'] + df['yeti_max_pct'] + df['drider_max_pct'] + \
    df['etitan_max_pct'] + df['root_max_pct'] + df['minion_max_pct'] + df['hog_max_pct'] + \
    df['valk_max_pct'] + df['golem_max_pct'] + df['witch_max_pct'] + df['lh_max_pct'] + \
    df['bowl_max_pct'] + df['ig_max_pct'] + df['hh_max_pct'] + df['aw_max_pct'] + \
    df['supbarb_max_pct'] + df['suparch_max_pct'] + df['sneakgob_max_pct'] + df['supwb_max_pct'] + \
    df['supgiant_max_pct'] + df['rockloon_max_pct'] + df['supwiz_max_pct'] + df['supdrag_max_pct'] + \
    df['infdrag_max_pct'] + df['supminion_max_pct'] + df['supvalk_max_pct'] + df['supwitch_max_pct'] + \
    df['ih_max_pct'] + df['supbowl_max_pct'] + df['supminer_max_pct'] + df['suphog_max_pct'] + \
    df['fc_max_pct'] + df['azdrag_max_pct']
   
df['cc_avg_below_max'] = df['cc_barb_from_max'] + df['cc_arch_from_max'] + \
    df['cc_giant_from_max'] + df['cc_gob_from_max'] + df['cc_wb_from_max'] + df['cc_wiz_from_max'] + df['cc_balloon_from_max'] + \
    df['cc_healer_from_max'] + df['cc_drag_from_max'] + df['cc_pekka_from_max'] + df['cc_babydrag_from_max'] + \
    df['cc_miner_from_max'] + df['cc_edrag_from_max'] + df['cc_yeti_from_max'] + df['cc_drider_from_max'] + \
    df['cc_etitan_from_max'] + df['cc_root_from_max'] + df['cc_minion_from_max'] + df['cc_hog_from_max'] + \
    df['cc_valk_from_max'] + df['cc_golem_from_max'] + df['cc_witch_from_max'] + df['cc_lh_from_max'] + \
    df['cc_bowl_from_max'] + df['cc_ig_from_max'] + df['cc_hh_from_max'] + df['cc_aw_from_max'] + \
    df['cc_supbarb_from_max'] + df['cc_suparch_from_max'] + df['cc_sneakgob_from_max'] + df['cc_supwb_from_max'] + \
    df['cc_supgiant_from_max'] + df['cc_rockloon_from_max'] + df['cc_supwiz_from_max'] + df['cc_supdrag_from_max'] + \
    df['cc_infdrag_from_max'] + df['cc_supminion_from_max'] + df['cc_supvalk_from_max'] + df['cc_supwitch_from_max'] + \
    df['cc_ih_from_max'] + df['cc_supbowl_from_max'] + df['cc_supminer_from_max'] + df['cc_suphog_from_max']

df['cc_max_percent'] = df['cc_barb_max_pct'] + df['cc_arch_max_pct'] + \
    df['cc_giant_max_pct'] + df['cc_gob_max_pct'] + df['cc_wb_max_pct'] + df['cc_wiz_max_pct'] + df['cc_balloon_max_pct'] + \
    df['cc_healer_max_pct'] + df['cc_drag_max_pct'] + df['cc_pekka_max_pct'] + df['cc_babydrag_max_pct'] + \
    df['cc_miner_max_pct'] + df['cc_edrag_max_pct'] + df['cc_yeti_max_pct'] + df['cc_drider_max_pct'] + \
    df['cc_etitan_max_pct'] + df['cc_root_max_pct'] + df['cc_minion_max_pct'] + df['cc_hog_max_pct'] + \
    df['cc_valk_max_pct'] + df['cc_golem_max_pct'] + df['cc_witch_max_pct'] + df['cc_lh_max_pct'] + \
    df['cc_bowl_max_pct'] + df['cc_ig_max_pct'] + df['cc_hh_max_pct'] + df['cc_aw_max_pct'] + \
    df['cc_supbarb_max_pct'] + df['cc_suparch_max_pct'] + df['cc_sneakgob_max_pct'] + df['cc_supwb_max_pct'] + \
    df['cc_supgiant_max_pct'] + df['cc_rockloon_max_pct'] + df['cc_supwiz_max_pct'] + df['cc_supdrag_max_pct'] + \
    df['cc_infdrag_max_pct'] + df['cc_supminion_max_pct'] + df['cc_supvalk_max_pct'] + df['cc_supwitch_max_pct'] + \
    df['cc_ih_max_pct'] + df['cc_supbowl_max_pct'] + df['cc_supminer_max_pct'] + df['cc_suphog_max_pct'] 

df['army_tot_avg_below_max'] = (df['army_size'] / df['army_size_total']) * df['army_avg_below_max'] + \
                                (df['cc_num_troops'] / df['army_size_total']) * df['cc_avg_below_max']

df['army_tot_max_percent'] = (df['army_size'] / df['army_size_total']) * df['army_max_percent'] + \
                                (df['cc_num_troops'] / df['army_size_total']) * df['cc_max_percent']

df.drop(df.filter(like='_from_max', axis=1), axis=1, inplace=True)
df.drop(df.filter(like='_max_pct', axis=1), axis=1, inplace=True)


# variables related to overall attack (th level, th location, base strength, etc.)
df['th_diff'] = df['th_attack'] - df['th_defend']
df['th_attack_higher'] = np.where(df['th_diff'] > 0, 1, 0)
df['th_attack_lower'] = np.where(df['th_diff'] < 0, 1, 0)
df['first_attack'] = np.where(df['base_attack_no'] == 1, 1, 0)
df['th_dist_x'] = df.apply(lambda row: min(45 - row['th_loc_x_defend'], row['th_loc_x_defend']), axis=1)
df['th_dist_y'] = df.apply(lambda row: min(45 - row['th_loc_y_defend'], row['th_loc_y_defend']), axis=1)
df['th_dist_total'] = np.sqrt(df['th_dist_x']**2 + df['th_dist_y']**2)
df['cwl'] = np.where(df['cwl_id'] == 0, 0, 1)

# engineering variables that are 0 if th are different levels
df['base_ww_same_th'] = np.where(df['th_diff'] == 0, df['base_ww_defend'], 0)
df['army_size_tot_same_th'] = np.where(df['th_diff'] == 0, df['army_size_total'], 0)
df['num_spells_same_th'] = np.where(df['th_diff'] == 0, df['num_spells'], 0)
df['th_dist_total_same_th'] = np.where(df['th_diff'] == 0, df['th_dist_total'], 0)
df['army_below_max_same_th'] = np.where(df['th_diff'] == 0, df['army_tot_avg_below_max'], 0)
df['spell_below_max_same_th'] = np.where(df['th_diff'] == 0, df['spell_avg_below_max'], 0)
df['sm_below_max_same_th'] = np.where(df['th_diff'] == 0, df['sm_below_max'], 0)
df['hero_num_same_th'] = np.where(df['th_diff'] == 0, df['hero_num'], 0)
df['hero_sum_same_th'] = np.where(df['th_diff'] == 0, df['hero_sum'], 0)
df['hero_total_strength_same_th'] = np.where(df['th_diff'] == 0, df['hero_total_strength'], 0)
df['pet_sum_same_th'] = np.where(df['th_diff'] == 0, df['pet_sum'], 0)
df['equip_sum_same_th'] = np.where(df['th_diff'] == 0, df['equip_sum'], 0)


# building indicator variables to identify attack strategies
df['strat_cookieram_50'] = np.where(df['cookie_pct'] + df['ram_pct'] > 0.6, 1, 0)
df['strat_cookieram_60'] = np.where(df['cookie_pct'] + df['ram_pct'] > 0.6, 1, 0)
df['strat_cookieram_70'] = np.where(df['cookie_pct'] + df['ram_pct'] > 0.7, 1, 0)
df['strat_edragloon'] = np.where(df['edrag_pct'] + df['balloon_pct'] > 0.8, 1, 0)
df['strat_lavaloon'] = np.where(((df['lh_num'] >= 1) | (df['ih_num'] >= 1)) & (df['balloon_pct'] > 0.2), 1, 0)
df['strat_air'] = np.where((df['air_pct'] - df['healer_pct']) / (1 - df['healer_pct']) > 0.75, 1, 0)
df['strat_ground'] = np.where(df['ground_pct'] / (1 - df['healer_pct']) > 0.75, 1, 0)
df['strat_ag_hybrid'] = np.where(((df['air_pct'] - df['healer_pct']) / (1 - df['healer_pct']) > 0.25) & \
                                    (df['ground_pct'] / (1 - df['healer_pct']) > 0.25), 1, 0)
df['strat_healers'] = np.where(df['healer_pct'] > 0.15, 1, 0)
df['strat_blizzard'] = np.where((df['bb_num'] == 1) & (df['invis_num'] > 0) & (df['cc_supwiz_pct'] + df['cc_suparch_pct'] > 0.5), 1, 0)

# create indicator variables for users
df['user_id_attack'] = df['user_id_attack'].astype(int)
unique_user_ids = df['user_id_attack'].unique()
for user_id in unique_user_ids:
    df[f'user_{user_id}'] = (df['user_id_attack'] == user_id).astype(int)

df['user_id_defend'] = df['user_id_defend'].astype(int)
unique_user_ids_def = df['user_id_defend'].unique()
for user_id in unique_user_ids_def:
    df[f'user_{user_id}_def'] = (df['user_id_defend'] == user_id).astype(int)

# create leg_trophy and war_stars variables as priors for user ability
min_attacks = 20
df['num_user_attacks'] = df.groupby('user_id_attack').transform('count').max(axis=1)

df['leg_trophy_prior'] = df['leg_trophy_attack'] * (df['num_user_attacks'] / min_attacks)
df['leg_trophy_adj'] = df['leg_trophy_attack'] - df['leg_trophy_prior']
df['leg_trophy_adj'] = np.where(df['leg_trophy_adj']  < 0, 0, df['leg_trophy_adj'])
df['leg_trophy_adj'] = np.where(df['user_id_attack'] == 0, df['leg_trophy_attack'], df['leg_trophy_adj'])

df['leg_trophy_mr_prior'] = df['leg_trophy_most_recent_attack'] * (df['num_user_attacks'] / min_attacks)
df['leg_trophy_mr_adj'] = df['leg_trophy_most_recent_attack'] - df['leg_trophy_mr_prior']
df['leg_trophy_mr_adj'] = np.where(df['leg_trophy_mr_adj']  < 0, 0, df['leg_trophy_mr_adj'])
df['leg_trophy_mr_adj'] = np.where(df['user_id_attack'] == 0, df['leg_trophy_most_recent_attack'], df['leg_trophy_mr_adj'])

df['war_stars_prior'] = df['war_stars_attack'] * (df['num_user_attacks'] / min_attacks)
df['war_stars_adj'] = df['war_stars_attack'] - df['war_stars_prior']
df['war_stars_adj'] = np.where(df['war_stars_adj']  < 0, 0, df['war_stars_adj'])
df['war_stars_adj'] = np.where(df['user_id_attack'] == 0, df['war_stars_attack'], df['war_stars_adj'])

# a couple metrics to check for mistakes in entries
df['army_check'] = df['barb_pct'] + df['ram_pct'] + df['cookie_pct'] + df['arch_pct'] + \
    df['giant_pct'] + df['gob_pct'] + df['wb_pct'] + df['wiz_pct'] + df['balloon_pct'] + \
    df['healer_pct'] + df['drag_pct'] + df['pekka_pct'] + df['babydrag_pct'] + \
    df['miner_pct'] + df['edrag_pct'] + df['yeti_pct'] + df['drider_pct'] + \
    df['etitan_pct'] + df['root_pct'] + df['minion_pct'] + df['hog_pct'] + \
    df['valk_pct'] + df['golem_pct'] + df['witch_pct'] + df['lh_pct'] + \
    df['bowl_pct'] + df['ig_pct'] + df['hh_pct'] + df['aw_pct'] + \
    df['supbarb_pct'] + df['suparch_pct'] + df['sneakgob_pct'] + df['supwb_pct'] + \
    df['supgiant_pct'] + df['rockloon_pct'] + df['supwiz_pct'] + df['supdrag_pct'] + \
    df['infdrag_pct'] + df['supminion_pct'] + df['supvalk_pct'] + df['supwitch_pct'] + \
    df['ih_pct'] + df['supbowl_pct'] + df['supminer_pct'] + df['suphog_pct'] + \
    df['fc_pct'] + df['azdrag_pct']
df['army_total_check'] = df['barb_pct_tot'] + df['ram_pct_tot'] + df['cookie_pct_tot'] + df['arch_pct_tot'] + \
    df['giant_pct_tot'] + df['gob_pct_tot'] + df['wb_pct_tot'] + df['wiz_pct_tot'] + df['balloon_pct_tot'] + \
    df['healer_pct_tot'] + df['drag_pct_tot'] + df['pekka_pct_tot'] + df['babydrag_pct_tot'] + \
    df['miner_pct_tot'] + df['edrag_pct_tot'] + df['yeti_pct_tot'] + df['drider_pct_tot'] + \
    df['etitan_pct_tot'] + df['root_pct_tot'] + df['minion_pct_tot'] + df['hog_pct_tot'] + \
    df['valk_pct_tot'] + df['golem_pct_tot'] + df['witch_pct_tot'] + df['lh_pct_tot'] + \
    df['bowl_pct_tot'] + df['ig_pct_tot'] + df['hh_pct_tot'] + df['aw_pct_tot'] + \
    df['supbarb_pct_tot'] + df['suparch_pct_tot'] + df['sneakgob_pct_tot'] + df['supwb_pct_tot'] + \
    df['supgiant_pct_tot'] + df['rockloon_pct_tot'] + df['supwiz_pct_tot'] + df['supdrag_pct_tot'] + \
    df['infdrag_pct_tot'] + df['supminion_pct_tot'] + df['supvalk_pct_tot'] + df['supwitch_pct_tot'] + \
    df['ih_pct_tot'] + df['supbowl_pct_tot'] + df['supminer_pct_tot'] + df['suphog_pct_tot'] + \
    df['fc_pct_tot'] + df['azdrag_pct_tot']
df['spell_check'] = df['light_pct'] + df['bof_pct'] + df['heal_pct'] + df['rage_pct'] + \
    df['jump_pct'] + df['freeze_pct'] + df['clone_pct'] + df['invis_pct'] + \
    df['recall_pct'] + df['poison_pct'] + df['eq_pct'] + df['haste_pct'] + \
    df['skel_pct'] + df['bat_pct'] + df['og_pct']
df['air_ground_check'] = df['air_pct'] + df['ground_pct']
df['army_size_diff'] = df['army_size'] - df['army_size_attack']


# save csv file with prepped data for model
if save_file == True:
    df.to_csv('attack_data_for_model.csv', index=False)

