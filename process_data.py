import pandas as pd
import ast


def process_data_GK(file):
    GK = pd.read_csv(file)
    GK['Attribute Vector'] = GK['Attribute Vector'].apply(ast.literal_eval)
    attribute_names = [
        'PSxG-GA', 'Goals Against', 'Save Percentage', 'PSxG/SoT',
        'Save% (Penalty Kicks)', 'Clean Sheet Percentage', 'Touches',
        'Launch %', 'Goal Kicks', 'Avg. Length of Goal Kicks',
        'Crosses Stopped %', 'Def. Actions Outside Pen. Area'
    ]
    new_GK = pd.DataFrame(columns=['Name'])
    for _, row in GK.iterrows():
        new_row = {'Name': row['Name']}
        for attribute_name, value in zip(attribute_names, row['Attribute Vector']):
            new_row[attribute_name] = value
        new_GK = pd.concat([new_GK, pd.DataFrame([new_row])])
    new_GK = new_GK.drop_duplicates(subset=['Name'])
    new_GK['Total Stats'] = new_GK[attribute_names].sum(axis=1)
    new_GK['Total Stats'] = new_GK['Total Stats'] - new_GK['Goals Against'] - new_GK['Avg. Length of Goal Kicks']
    features = new_GK[attribute_names].values
    target = new_GK['Total Stats'].values
    player_names = new_GK['Name'].values
    return features, target, player_names


def process_data_DC(file):
    DC = pd.read_csv(file)
    DC['Attribute Vector'] = DC['Attribute Vector'].apply(ast.literal_eval)

    new_attribute_names = [
        'Non-Penalty Goals', 'Non-Penalty xG', 'Shots Total', 'Assists',
        'xAG', 'npxG + xAG', 'Shot-Creating Actions', 'Passes Attempted',
        'Pass Completion %', 'Progressive Passes', 'Progressive Carries',
        'Successful Take-Ons', 'Touches (Att Pen)', 'Progressive Passes Rec',
        'Tackles', 'Interceptions', 'Blocks', 'Clearances', 'Aerials won'
    ]

    new_DC = pd.DataFrame(columns=['Name'])

    for _, row in DC.iterrows():
        new_row = {'Name': row['Name']}
        for attribute_name, value in zip(new_attribute_names, row['Attribute Vector']):
            new_row[attribute_name] = value
        new_DC = pd.concat([new_DC, pd.DataFrame([new_row])])

    attribute_names = ['Tackles', 'Interceptions', 'Blocks', 'Clearances', 'Aerials won', 'Pass Completion %']
    new_DC = new_DC.drop_duplicates(subset=['Name'])
    new_DC['Total Stats'] = new_DC[attribute_names].sum(axis=1)
    new_DC = new_DC[['Name'] + attribute_names + ['Total Stats']]
    features = new_DC[attribute_names].values
    target = new_DC['Total Stats'].values
    player_names = new_DC['Name'].values
    return features, target, player_names


def process_data_FB(file):
    FB = pd.read_csv(file)

    FB['Attribute Vector'] = FB['Attribute Vector'].apply(ast.literal_eval)

    new_attribute_names = [
        'Non-Penalty Goals', 'Non-Penalty xG', 'Shots Total', 'Assists',
        'xAG', 'npxG + xAG', 'Shot-Creating Actions', 'Passes Attempted',
        'Pass Completion %', 'Progressive Passes', 'Progressive Carries',
        'Successful Take-Ons', 'Touches (Att Pen)', 'Progressive Passes Rec',
        'Tackles', 'Interceptions', 'Blocks', 'Clearances', 'Aerials won'
    ]

    new_FB = pd.DataFrame(columns=['Name'])

    for _, row in FB.iterrows():
        new_row = {'Name': row['Name']}
        for attribute_name, value in zip(new_attribute_names, row['Attribute Vector']):
            new_row[attribute_name] = value
        new_FB = pd.concat([new_FB, pd.DataFrame([new_row])])

    attribute_names = ['npxG + xAG', 'Tackles', 'Interceptions', 'Blocks', 'Clearances', 'Aerials won',
                       'Pass Completion %']
    new_FB = new_FB.drop_duplicates(subset=['Name'])
    new_FB['Total Stats'] = new_FB[attribute_names].sum(axis=1)

    new_FB = new_FB[['Name'] + attribute_names + ['Total Stats']]
    features = new_FB[attribute_names].values
    target = new_FB['Total Stats'].values
    player_names = new_FB['Name'].values
    return features, target, player_names


def process_data_MD(file):
    MD = pd.read_csv(file)

    MD['Attribute Vector'] = MD['Attribute Vector'].apply(ast.literal_eval)

    new_attribute_names = [
        'Non-Penalty Goals', 'Non-Penalty xG', 'Shots Total', 'Assists',
        'xAG', 'npxG + xAG', 'Shot-Creating Actions', 'Passes Attempted',
        'Pass Completion %', 'Progressive Passes', 'Progressive Carries',
        'Successful Take-Ons', 'Touches (Att Pen)', 'Progressive Passes Rec',
        'Tackles', 'Interceptions', 'Blocks', 'Clearances', 'Aerials won'
    ]

    new_MD = pd.DataFrame(columns=['Name'])

    for _, row in MD.iterrows():
        new_row = {'Name': row['Name']}
        for attribute_name, value in zip(new_attribute_names, row['Attribute Vector']):
            new_row[attribute_name] = value
        new_MD = pd.concat([new_MD, pd.DataFrame([new_row])])

    attribute_names = ['Tackles', 'Interceptions', 'npxG + xAG', 'Shot-Creating Actions', 'Passes Attempted',
                       'Pass Completion %', 'Progressive Passes', 'Progressive Carries',
                       'Successful Take-Ons']
    new_MD = new_MD.drop_duplicates(subset=['Name'])
    new_MD['Total Stats'] = new_MD[attribute_names].sum(axis=1)

    new_MD = new_MD[['Name'] + attribute_names + ['Total Stats']]

    features = new_MD[attribute_names].values
    target = new_MD['Total Stats'].values
    player_names = new_MD['Name'].values
    return features, target, player_names


def process_data_WG(file):
    WG = pd.read_csv(file)

    WG['Attribute Vector'] = WG['Attribute Vector'].apply(ast.literal_eval)

    new_attribute_names = [
        'Non-Penalty Goals', 'Non-Penalty xG', 'Shots Total', 'Assists',
        'xAG', 'npxG + xAG', 'Shot-Creating Actions', 'Passes Attempted',
        'Pass Completion %', 'Progressive Passes', 'Progressive Carries',
        'Successful Take-Ons', 'Touches (Att Pen)', 'Progressive Passes Rec',
        'Tackles', 'Interceptions', 'Blocks', 'Clearances', 'Aerials won'
    ]

    new_WG = pd.DataFrame(columns=['Name'])

    for _, row in WG.iterrows():
        new_row = {'Name': row['Name']}
        for attribute_name, value in zip(new_attribute_names, row['Attribute Vector']):
            new_row[attribute_name] = value
        new_WG = pd.concat([new_WG, pd.DataFrame([new_row])])

    attribute_names = ['Non-Penalty Goals', 'Non-Penalty xG', 'Shots Total', 'Assists',
                       'xAG', 'npxG + xAG', 'Shot-Creating Actions', 'Passes Attempted',
                       'Pass Completion %', 'Progressive Passes', 'Progressive Carries',
                       'Successful Take-Ons']
    new_WG = new_WG.drop_duplicates(subset=['Name'])
    new_WG['Total Stats'] = new_WG[attribute_names].sum(axis=1)

    new_WG = new_WG[['Name'] + attribute_names + ['Total Stats']]
    features = new_WG[attribute_names].values
    target = new_WG['Total Stats'].values
    player_names = new_WG['Name'].values
    return features, target, player_names


def process_data_FW(file):
    FW = pd.read_csv(file)

    FW['Attribute Vector'] = FW['Attribute Vector'].apply(ast.literal_eval)

    new_attribute_names = [
        'Non-Penalty Goals', 'Non-Penalty xG', 'Shots Total', 'Assists',
        'xAG', 'npxG + xAG', 'Shot-Creating Actions', 'Passes Attempted',
        'Pass Completion %', 'Progressive Passes', 'Progressive Carries',
        'Successful Take-Ons', 'Touches (Att Pen)', 'Progressive Passes Rec',
        'Tackles', 'Interceptions', 'Blocks', 'Clearances', 'Aerials won'
    ]

    new_FW = pd.DataFrame(columns=['Name'])

    for _, row in FW.iterrows():
        new_row = {'Name': row['Name']}
        for attribute_name, value in zip(new_attribute_names, row['Attribute Vector']):
            new_row[attribute_name] = value
        new_FW = pd.concat([new_FW, pd.DataFrame([new_row])])

    attribute_names = ['Non-Penalty Goals', 'Non-Penalty xG', 'Shots Total', 'Assists', 'xAG', 'npxG + xAG',
                       'Shot-Creating Actions']
    new_FW = new_FW.drop_duplicates(subset=['Name'])
    new_FW['Total Stats'] = new_FW[attribute_names].sum(axis=1)

    new_FW = new_FW[['Name'] + attribute_names + ['Total Stats']]

    features = new_FW[attribute_names].values
    target = new_FW['Total Stats'].values
    player_names = new_FW['Name'].values
    return features, target, player_names
