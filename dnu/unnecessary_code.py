#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:55:10 2024

@author: ryandrost
"""

scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a softmax regression model
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Train the model
softmax_model.fit(X_train2, y_train)

# Make predictions on the test set
y_pred = softmax_model.predict(X_train2)

# Evaluate the accuracy
accuracy = accuracy_score(y_train, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# You can also access the learned coefficients and intercepts
print('Coefficients:', softmax_model.coef_)
print('Intercepts:', softmax_model.intercept_)

conf_matrix = confusion_matrix(y_train, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

'''df.drop(['bkequip1', 'bkequip2', 'bkequip1_lvl', 'bkequip2_lvl', 'bkpet',
          'aqequip1', 'aqequip2', 'aqequip1_lvl', 'aqequip2_lvl', 'aqpet',
          'gwequip1', 'gwequip2', 'gwequip1_lvl', 'gwequip2_lvl', 'gwpet',
          'rcequip1', 'rcequip2', 'rcequip1_lvl', 'rcequip2_lvl', 'rcpet',
          'troops', 'spells', 'cc_troops', 'cc_spells', '0_lvl'], axis = 1, inplace=True)'''

'''df['light_tot'] = df['light_num'] + df['cc_light_num']
df['bof_tot'] = df['bof_num'] + df['cc_bof_num']
df['heal_tot'] = df['heal_num'] + df['cc_heal_num']
df['rage_tot'] = df['rage_num'] + df['cc_rage_num']
df['jump_tot'] = df['jump_num'] + df['cc_jump_num']
df['freeze_tot'] = df['freeze_num'] + df['cc_freeze_num']
df['clone_tot'] = df['clone_num'] + df['cc_clone_num']
df['invis_tot'] = df['invis_num'] + df['cc_invis_num']
df['recall_tot'] = df['recall_num'] + df['cc_recall_num']
df['poison_tot'] = df['poison_num'] + df['cc_poison_num']
df['eq_tot'] = df['eq_num'] + df['cc_eq_num']
df['haste_tot'] = df['haste_num'] + df['cc_haste_num']
df['skel_tot'] = df['skel_num'] + df['cc_skel_num']
df['bat_tot'] = df['bat_num'] + df['cc_bat_num']

df['light_space'] = int(unit_info[unit_info['unit'] == 'light']['housing'])*df['light_tot']
df['bof_space'] = int(unit_info[unit_info['unit'] == 'bof']['housing'])*df['bof_tot']
df['heal_space'] = int(unit_info[unit_info['unit'] == 'heal']['housing'])*df['heal_tot']
df['rage_space'] = int(unit_info[unit_info['unit'] == 'rage']['housing'])*df['rage_tot']
df['jump_space'] = int(unit_info[unit_info['unit'] == 'jump']['housing'])*df['jump_tot']
df['freeze_space'] = int(unit_info[unit_info['unit'] == 'freeze']['housing'])*df['freeze_tot']
df['clone_space'] = int(unit_info[unit_info['unit'] == 'clone']['housing'])*df['clone_tot']
df['invis_space'] = int(unit_info[unit_info['unit'] == 'invis']['housing'])*df['invis_tot']
df['recall_space'] = int(unit_info[unit_info['unit'] == 'recall']['housing'])*df['recall_tot']
df['poison_space'] = int(unit_info[unit_info['unit'] == 'poison']['housing'])*df['poison_tot']
df['eq_space'] = int(unit_info[unit_info['unit'] == 'eq']['housing'])*df['eq_tot']
df['haste_space'] = int(unit_info[unit_info['unit'] == 'haste']['housing'])*df['haste_tot']
df['skel_space'] = int(unit_info[unit_info['unit'] == 'skel']['housing'])*df['skel_tot']
df['bat_space'] = int(unit_info[unit_info['unit'] == 'bat']['housing'])*df['bat_tot']


df['light_pct'] = df['light_space'] / df['num_spells']
df['bof_pct'] = df['bof_space'] / df['num_spells']
df['heal_pct'] = df['heal_space'] / df['num_spells']
df['rage_pct'] = df['rage_space'] / df['num_spells']
df['jump_pct'] = df['jump_space'] / df['num_spells']
df['freeze_pct'] = df['freeze_space'] / df['num_spells']
df['clone_pct'] = df['clone_space'] / df['num_spells']
df['invis_pct'] = df['invis_space'] / df['num_spells']
df['recall_pct'] = df['recall_space'] / df['num_spells']
df['poison_pct'] = df['poison_space'] / df['num_spells']
df['eq_pct'] = df['eq_space'] / df['num_spells']
df['haste_pct'] = df['haste_space'] / df['num_spells']
df['skel_pct'] = df['skel_space'] / df['num_spells']
df['bat_pct'] = df['bat_space'] / df['num_spells']

df['barb_space'] = int(unit_info[unit_info['unit'] == 'barb']['housing'])*df['barb_num']
df['ram_space'] = int(unit_info[unit_info['unit'] == 'ram']['housing'])*df['ram_num']
df['cookie_space'] = int(unit_info[unit_info['unit'] == 'cookie']['housing'])*df['cookie_num']
df['arch_space'] = int(unit_info[unit_info['unit'] == 'arch']['housing'])*df['arch_num']
df['giant_space'] = int(unit_info[unit_info['unit'] == 'giant']['housing'])*df['giant_num']
df['gob_space'] = int(unit_info[unit_info['unit'] == 'gob']['housing'])*df['gob_num']
df['wb_space'] = int(unit_info[unit_info['unit'] == 'wb']['housing'])*df['wb_num']
df['wiz_space'] = int(unit_info[unit_info['unit'] == 'wiz']['housing'])*df['wiz_num']
df['balloon_space'] = int(unit_info[unit_info['unit'] == 'balloon']['housing'])*df['balloon_num']
df['healer_space'] = int(unit_info[unit_info['unit'] == 'healer']['housing'])*df['healer_num']
df['drag_space'] = int(unit_info[unit_info['unit'] == 'drag']['housing'])*df['drag_num']
df['pekka_space'] = int(unit_info[unit_info['unit'] == 'pekka']['housing'])*df['pekka_num']
df['babydrag_space'] = int(unit_info[unit_info['unit'] == 'babydrag']['housing'])*df['babydrag_num']
df['miner_space'] = int(unit_info[unit_info['unit'] == 'miner']['housing'])*df['miner_num']
df['edrag_space'] = int(unit_info[unit_info['unit'] == 'edrag']['housing'])*df['edrag_num']
df['yeti_space'] = int(unit_info[unit_info['unit'] == 'yeti']['housing'])*df['yeti_num']
df['drider_space'] = int(unit_info[unit_info['unit'] == 'drider']['housing'])*df['drider_num']
df['etitan_space'] = int(unit_info[unit_info['unit'] == 'etitan']['housing'])*df['etitan_num']
df['root_space'] = int(unit_info[unit_info['unit'] == 'root']['housing'])*df['root_num']
df['minion_space'] = int(unit_info[unit_info['unit'] == 'minion']['housing'])*df['minion_num']
df['hog_space'] = int(unit_info[unit_info['unit'] == 'hog']['housing'])*df['hog_num']
df['valk_space'] = int(unit_info[unit_info['unit'] == 'valk']['housing'])*df['valk_num']
df['golem_space'] = int(unit_info[unit_info['unit'] == 'golem']['housing'])*df['golem_num']
df['witch_space'] = int(unit_info[unit_info['unit'] == 'witch']['housing'])*df['witch_num']
df['lh_space'] = int(unit_info[unit_info['unit'] == 'lh']['housing'])*df['lh_num']
df['bowl_space'] = int(unit_info[unit_info['unit'] == 'bowl']['housing'])*df['bowl_num']
df['ig_space'] = int(unit_info[unit_info['unit'] == 'ig']['housing'])*df['ig_num']
df['hh_space'] = int(unit_info[unit_info['unit'] == 'hh']['housing'])*df['hh_num']
df['aw_space'] = int(unit_info[unit_info['unit'] == 'aw']['housing'])*df['aw_num']
df['supbarb_space'] = int(unit_info[unit_info['unit'] == 'supbarb']['housing'])*df['supbarb_num']
df['suparch_space'] = int(unit_info[unit_info['unit'] == 'suparch']['housing'])*df['suparch_num']
df['sneakgob_space'] = int(unit_info[unit_info['unit'] == 'sneakgob']['housing'])*df['sneakgob_num']
df['supwb_space'] = int(unit_info[unit_info['unit'] == 'supwb']['housing'])*df['supwb_num']
df['supgiant_space'] = int(unit_info[unit_info['unit'] == 'supgiant']['housing'])*df['supgiant_num']
df['rockloon_space'] = int(unit_info[unit_info['unit'] == 'rockloon']['housing'])*df['rockloon_num']
df['supwiz_space'] = int(unit_info[unit_info['unit'] == 'supwiz']['housing'])*df['supwiz_num']
df['supdrag_space'] = int(unit_info[unit_info['unit'] == 'supdrag']['housing'])*df['supdrag_num']
df['infdrag_space'] = int(unit_info[unit_info['unit'] == 'infdrag']['housing'])*df['infdrag_num']
df['supminion_space'] = int(unit_info[unit_info['unit'] == 'supminion']['housing'])*df['supminion_num']
df['supvalk_space'] = int(unit_info[unit_info['unit'] == 'supvalk']['housing'])*df['supvalk_num']
df['supwitch_space'] = int(unit_info[unit_info['unit'] == 'supwitch']['housing'])*df['supwitch_num']
df['ih_space'] = int(unit_info[unit_info['unit'] == 'ih']['housing'])*df['ih_num']
df['supbowl_space'] = int(unit_info[unit_info['unit'] == 'supbowl']['housing'])*df['supbowl_num']
df['supminer_space'] = int(unit_info[unit_info['unit'] == 'supminer']['housing'])*df['supminer_num']
df['suphog_space'] = int(unit_info[unit_info['unit'] == 'suphog']['housing'])*df['suphog_num']

df['barb_pct'] = df['barb_space'] / df['army_size']
df['ram_pct'] = df['ram_space'] / df['army_size']
df['cookie_pct'] = df['cookie_space'] / df['army_size']
df['arch_pct'] = df['arch_space'] / df['army_size']
df['giant_pct'] = df['giant_space'] / df['army_size']
df['gob_pct'] = df['gob_space'] / df['army_size']
df['wb_pct'] = df['wb_space'] / df['army_size']
df['wiz_pct'] = df['wiz_space'] / df['army_size']
df['balloon_pct'] = df['balloon_space'] / df['army_size']
df['healer_pct'] = df['healer_space'] / df['army_size']
df['drag_pct'] = df['drag_space'] / df['army_size']
df['pekka_pct'] = df['pekka_space'] / df['army_size']
df['babydrag_pct'] = df['babydrag_space'] / df['army_size']
df['miner_pct'] = df['miner_space'] / df['army_size']
df['edrag_pct'] = df['edrag_space'] / df['army_size']
df['yeti_pct'] = df['yeti_space'] / df['army_size']
df['drider_pct'] = df['drider_space'] / df['army_size']
df['etitan_pct'] = df['etitan_space'] / df['army_size']
df['root_pct'] = df['root_space'] / df['army_size']
df['minion_pct'] = df['minion_space'] / df['army_size']
df['hog_pct'] = df['hog_space'] / df['army_size']
df['valk_pct'] = df['valk_space'] / df['army_size']
df['golem_pct'] = df['golem_space'] / df['army_size']
df['witch_pct'] = df['witch_space'] / df['army_size']
df['lh_pct'] = df['lh_space'] / df['army_size']
df['bowl_pct'] = df['bowl_space'] / df['army_size']
df['ig_pct'] = df['ig_space'] / df['army_size']
df['hh_pct'] = df['hh_space'] / df['army_size']
df['aw_pct'] = df['aw_space'] / df['army_size']
df['supbarb_pct'] = df['supbarb_space'] / df['army_size']
df['suparch_pct'] = df['suparch_space'] / df['army_size']
df['sneakgob_pct'] = df['sneakgob_space'] / df['army_size']
df['supwb_pct'] = df['supwb_space'] / df['army_size']
df['supgiant_pct'] = df['supgiant_space'] / df['army_size']
df['rockloon_pct'] = df['rockloon_space'] / df['army_size']
df['supwiz_pct'] = df['supwiz_space'] / df['army_size']
df['supdrag_pct'] = df['supdrag_space'] / df['army_size']
df['infdrag_pct'] = df['infdrag_space'] / df['army_size']
df['supminion_pct'] = df['supminion_space'] / df['army_size']
df['supvalk_pct'] = df['supvalk_space'] / df['army_size']
df['supwitch_pct'] = df['supwitch_space'] / df['army_size']
df['ih_pct'] = df['ih_space'] / df['army_size']
df['supbowl_pct'] = df['supbowl_space'] / df['army_size']
df['supminer_pct'] = df['supminer_space'] / df['army_size']
df['suphog_pct'] = df['suphog_space'] / df['army_size']

df['cc_barb_space'] = int(unit_info[unit_info['unit'] == 'barb']['housing'])*df['cc_barb_num']
df['cc_ram_space'] = int(unit_info[unit_info['unit'] == 'ram']['housing'])*df['cc_ram_num']
df['cc_cookie_space'] = int(unit_info[unit_info['unit'] == 'cookie']['housing'])*df['cc_cookie_num']
df['cc_arch_space'] = int(unit_info[unit_info['unit'] == 'arch']['housing'])*df['cc_arch_num']
df['cc_giant_space'] = int(unit_info[unit_info['unit'] == 'giant']['housing'])*df['cc_giant_num']
df['cc_gob_space'] = int(unit_info[unit_info['unit'] == 'gob']['housing'])*df['cc_gob_num']
df['cc_wb_space'] = int(unit_info[unit_info['unit'] == 'wb']['housing'])*df['cc_wb_num']
df['cc_wiz_space'] = int(unit_info[unit_info['unit'] == 'wiz']['housing'])*df['cc_wiz_num']
df['cc_balloon_space'] = int(unit_info[unit_info['unit'] == 'balloon']['housing'])*df['cc_balloon_num']
df['cc_healer_space'] = int(unit_info[unit_info['unit'] == 'healer']['housing'])*df['cc_healer_num']
df['cc_drag_space'] = int(unit_info[unit_info['unit'] == 'drag']['housing'])*df['cc_drag_num']
df['cc_pekka_space'] = int(unit_info[unit_info['unit'] == 'pekka']['housing'])*df['cc_pekka_num']
df['cc_babydrag_space'] = int(unit_info[unit_info['unit'] == 'babydrag']['housing'])*df['cc_babydrag_num']
df['cc_miner_space'] = int(unit_info[unit_info['unit'] == 'miner']['housing'])*df['cc_miner_num']
df['cc_edrag_space'] = int(unit_info[unit_info['unit'] == 'edrag']['housing'])*df['cc_edrag_num']
df['cc_yeti_space'] = int(unit_info[unit_info['unit'] == 'yeti']['housing'])*df['cc_yeti_num']
df['cc_drider_space'] = int(unit_info[unit_info['unit'] == 'drider']['housing'])*df['cc_drider_num']
df['cc_etitan_space'] = int(unit_info[unit_info['unit'] == 'etitan']['housing'])*df['cc_etitan_num']
df['cc_root_space'] = int(unit_info[unit_info['unit'] == 'root']['housing'])*df['cc_root_num']
df['cc_minion_space'] = int(unit_info[unit_info['unit'] == 'minion']['housing'])*df['cc_minion_num']
df['cc_hog_space'] = int(unit_info[unit_info['unit'] == 'hog']['housing'])*df['cc_hog_num']
df['cc_valk_space'] = int(unit_info[unit_info['unit'] == 'valk']['housing'])*df['cc_valk_num']
df['cc_golem_space'] = int(unit_info[unit_info['unit'] == 'golem']['housing'])*df['cc_golem_num']
df['cc_witch_space'] = int(unit_info[unit_info['unit'] == 'witch']['housing'])*df['cc_witch_num']
df['cc_lh_space'] = int(unit_info[unit_info['unit'] == 'lh']['housing'])*df['cc_lh_num']
df['cc_bowl_space'] = int(unit_info[unit_info['unit'] == 'bowl']['housing'])*df['cc_bowl_num']
df['cc_ig_space'] = int(unit_info[unit_info['unit'] == 'ig']['housing'])*df['cc_ig_num']
df['cc_hh_space'] = int(unit_info[unit_info['unit'] == 'hh']['housing'])*df['cc_hh_num']
df['cc_aw_space'] = int(unit_info[unit_info['unit'] == 'aw']['housing'])*df['cc_aw_num']
df['cc_supbarb_space'] = int(unit_info[unit_info['unit'] == 'supbarb']['housing'])*df['cc_supbarb_num']
df['cc_suparch_space'] = int(unit_info[unit_info['unit'] == 'suparch']['housing'])*df['cc_suparch_num']
df['cc_sneakgob_space'] = int(unit_info[unit_info['unit'] == 'sneakgob']['housing'])*df['cc_sneakgob_num']
df['cc_supwb_space'] = int(unit_info[unit_info['unit'] == 'supwb']['housing'])*df['cc_supwb_num']
df['cc_supgiant_space'] = int(unit_info[unit_info['unit'] == 'supgiant']['housing'])*df['cc_supgiant_num']
df['cc_rockloon_space'] = int(unit_info[unit_info['unit'] == 'rockloon']['housing'])*df['cc_rockloon_num']
df['cc_supwiz_space'] = int(unit_info[unit_info['unit'] == 'supwiz']['housing'])*df['cc_supwiz_num']
df['cc_supdrag_space'] = int(unit_info[unit_info['unit'] == 'supdrag']['housing'])*df['cc_supdrag_num']
df['cc_infdrag_space'] = int(unit_info[unit_info['unit'] == 'infdrag']['housing'])*df['cc_infdrag_num']
df['cc_supminion_space'] = int(unit_info[unit_info['unit'] == 'supminion']['housing'])*df['cc_supminion_num']
df['cc_supvalk_space'] = int(unit_info[unit_info['unit'] == 'supvalk']['housing'])*df['cc_supvalk_num']
df['cc_supwitch_space'] = int(unit_info[unit_info['unit'] == 'supwitch']['housing'])*df['cc_supwitch_num']
df['cc_ih_space'] = int(unit_info[unit_info['unit'] == 'ih']['housing'])*df['cc_ih_num']
df['cc_supbowl_space'] = int(unit_info[unit_info['unit'] == 'supbowl']['housing'])*df['cc_supbowl_num']
df['cc_supminer_space'] = int(unit_info[unit_info['unit'] == 'supminer']['housing'])*df['cc_supminer_num']
df['cc_suphog_space'] = int(unit_info[unit_info['unit'] == 'suphog']['housing'])*df['cc_suphog_num']

df['cc_barb_pct'] = df['cc_barb_space'] / df['cc_num_troops']
df['cc_ram_pct'] = df['cc_ram_space'] / df['cc_num_troops']
df['cc_cookie_pct'] = df['cc_cookie_space'] / df['cc_num_troops']
df['cc_arch_pct'] = df['cc_arch_space'] / df['cc_num_troops']
df['cc_giant_pct'] = df['cc_giant_space'] / df['cc_num_troops']
df['cc_gob_pct'] = df['cc_gob_space'] / df['cc_num_troops']
df['cc_wb_pct'] = df['cc_wb_space'] / df['cc_num_troops']
df['cc_wiz_pct'] = df['cc_wiz_space'] / df['cc_num_troops']
df['cc_balloon_pct'] = df['cc_balloon_space'] / df['cc_num_troops']
df['cc_healer_pct'] = df['cc_healer_space'] / df['cc_num_troops']
df['cc_drag_pct'] = df['cc_drag_space'] / df['cc_num_troops']
df['cc_pekka_pct'] = df['cc_pekka_space'] / df['cc_num_troops']
df['cc_babydrag_pct'] = df['cc_babydrag_space'] / df['cc_num_troops']
df['cc_miner_pct'] = df['cc_miner_space'] / df['cc_num_troops']
df['cc_edrag_pct'] = df['cc_edrag_space'] / df['cc_num_troops']
df['cc_yeti_pct'] = df['cc_yeti_space'] / df['cc_num_troops']
df['cc_drider_pct'] = df['cc_drider_space'] / df['cc_num_troops']
df['cc_etitan_pct'] = df['cc_etitan_space'] / df['cc_num_troops']
df['cc_root_pct'] = df['cc_root_space'] / df['cc_num_troops']
df['cc_minion_pct'] = df['cc_minion_space'] / df['cc_num_troops']
df['cc_hog_pct'] = df['cc_hog_space'] / df['cc_num_troops']
df['cc_valk_pct'] = df['cc_valk_space'] / df['cc_num_troops']
df['cc_golem_pct'] = df['cc_golem_space'] / df['cc_num_troops']
df['cc_witch_pct'] = df['cc_witch_space'] / df['cc_num_troops']
df['cc_lh_pct'] = df['cc_lh_space'] / df['cc_num_troops']
df['cc_bowl_pct'] = df['cc_bowl_space'] / df['cc_num_troops']
df['cc_ig_pct'] = df['cc_ig_space'] / df['cc_num_troops']
df['cc_hh_pct'] = df['cc_hh_space'] / df['cc_num_troops']
df['cc_aw_pct'] = df['cc_aw_space'] / df['cc_num_troops']
df['cc_supbarb_pct'] = df['cc_supbarb_space'] / df['cc_num_troops']
df['cc_suparch_pct'] = df['cc_suparch_space'] / df['cc_num_troops']
df['cc_sneakgob_pct'] = df['cc_sneakgob_space'] / df['cc_num_troops']
df['cc_supwb_pct'] = df['cc_supwb_space'] / df['cc_num_troops']
df['cc_supgiant_pct'] = df['cc_supgiant_space'] / df['cc_num_troops']
df['cc_rockloon_pct'] = df['cc_rockloon_space'] / df['cc_num_troops']
df['cc_supwiz_pct'] = df['cc_supwiz_space'] / df['cc_num_troops']
df['cc_supdrag_pct'] = df['cc_supdrag_space'] / df['cc_num_troops']
df['cc_infdrag_pct'] = df['cc_infdrag_space'] / df['cc_num_troops']
df['cc_supminion_pct'] = df['cc_supminion_space'] / df['cc_num_troops']
df['cc_supvalk_pct'] = df['cc_supvalk_space'] / df['cc_num_troops']
df['cc_supwitch_pct'] = df['cc_supwitch_space'] / df['cc_num_troops']
df['cc_ih_pct'] = df['cc_ih_space'] / df['cc_num_troops']
df['cc_supbowl_pct'] = df['cc_supbowl_space'] / df['cc_num_troops']
df['cc_supminer_pct'] = df['cc_supminer_space'] / df['cc_num_troops']
df['cc_suphog_pct'] = df['cc_suphog_space'] / df['cc_num_troops']

df['barb_pct_tot'] = (df['barb_space'] + df['cc_barb_space']) / df['army_size_total']
df['ram_pct_tot'] = (df['ram_space'] + df['cc_ram_space']) / df['army_size_total']
df['cookie_pct_tot'] = (df['cookie_space'] + df['cc_cookie_space']) / df['army_size_total']
df['arch_pct_tot'] = (df['arch_space'] + df['cc_arch_space']) / df['army_size_total']
df['giant_pct_tot'] = (df['giant_space'] + df['cc_giant_space']) / df['army_size_total']
df['gob_pct_tot'] = (df['gob_space'] + df['cc_gob_space']) / df['army_size_total']
df['wb_pct_tot'] = (df['wb_space'] + df['cc_wb_space']) / df['army_size_total']
df['wiz_pct_tot'] = (df['wiz_space'] + df['cc_wiz_space']) / df['army_size_total']
df['balloon_pct_tot'] = (df['balloon_space'] + df['cc_balloon_space']) / df['army_size_total']
df['healer_pct_tot'] = (df['healer_space'] + df['cc_healer_space']) / df['army_size_total']
df['drag_pct_tot'] = (df['drag_space'] + df['cc_drag_space']) / df['army_size_total']
df['pekka_pct_tot'] = (df['pekka_space'] + df['cc_pekka_space']) / df['army_size_total']
df['babydrag_pct_tot'] = (df['babydrag_space'] + df['cc_babydrag_space']) / df['army_size_total']
df['miner_pct_tot'] = (df['miner_space'] + df['cc_miner_space']) / df['army_size_total']
df['edrag_pct_tot'] = (df['edrag_space'] + df['cc_edrag_space']) / df['army_size_total']
df['yeti_pct_tot'] = (df['yeti_space'] + df['cc_yeti_space']) / df['army_size_total']
df['drider_pct_tot'] = (df['drider_space'] + df['cc_drider_space']) / df['army_size_total']
df['etitan_pct_tot'] = (df['etitan_space'] + df['cc_etitan_space']) / df['army_size_total']
df['root_pct_tot'] = (df['root_space'] + df['cc_root_space']) / df['army_size_total']
df['minion_pct_tot'] = (df['minion_space'] + df['cc_minion_space']) / df['army_size_total']
df['hog_pct_tot'] = (df['hog_space'] + df['cc_hog_space']) / df['army_size_total']
df['valk_pct_tot'] = (df['valk_space'] + df['cc_valk_space']) / df['army_size_total']
df['golem_pct_tot'] = (df['golem_space'] + df['cc_golem_space']) / df['army_size_total']
df['witch_pct_tot'] = (df['witch_space'] + df['cc_witch_space']) / df['army_size_total']
df['lh_pct_tot'] = (df['lh_space'] + df['cc_lh_space']) / df['army_size_total']
df['bowl_pct_tot'] = (df['bowl_space'] + df['cc_bowl_space']) / df['army_size_total']
df['ig_pct_tot'] = (df['ig_space'] + df['cc_ig_space']) / df['army_size_total']
df['hh_pct_tot'] = (df['hh_space'] + df['cc_hh_space']) / df['army_size_total']
df['aw_pct_tot'] = (df['aw_space'] + df['cc_aw_space']) / df['army_size_total']
df['supbarb_pct_tot'] = (df['supbarb_space'] + df['cc_supbarb_space']) / df['army_size_total']
df['suparch_pct_tot'] = (df['suparch_space'] + df['cc_suparch_space']) / df['army_size_total']
df['sneakgob_pct_tot'] = (df['sneakgob_space'] + df['cc_sneakgob_space']) / df['army_size_total']
df['supwb_pct_tot'] = (df['supwb_space'] + df['cc_supwb_space']) / df['army_size_total']
df['supgiant_pct_tot'] = (df['supgiant_space'] + df['cc_supgiant_space']) / df['army_size_total']
df['rockloon_pct_tot'] = (df['rockloon_space'] + df['cc_rockloon_space']) / df['army_size_total']
df['supwiz_pct_tot'] = (df['supwiz_space'] + df['cc_supwiz_space']) / df['army_size_total']
df['supdrag_pct_tot'] = (df['supdrag_space'] + df['cc_supdrag_space']) / df['army_size_total']
df['infdrag_pct_tot'] = (df['infdrag_space'] + df['cc_infdrag_space']) / df['army_size_total']
df['supminion_pct_tot'] = (df['supminion_space'] + df['cc_supminion_space']) / df['army_size_total']
df['supvalk_pct_tot'] = (df['supvalk_space'] + df['cc_supvalk_space']) / df['army_size_total']
df['supwitch_pct_tot'] = (df['supwitch_space'] + df['cc_supwitch_space']) / df['army_size_total']
df['ih_pct_tot'] = (df['ih_space'] + df['cc_ih_space']) / df['army_size_total']
df['supbowl_pct_tot'] = (df['supbowl_space'] + df['cc_supbowl_space']) / df['army_size_total']
df['supminer_pct_tot'] = (df['supminer_space'] + df['cc_supminer_space']) / df['army_size_total']
df['suphog_pct_tot'] = (df['suphog_space'] + df['cc_suphog_space']) / df['army_size_total']'''