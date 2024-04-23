from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from func import load_model, load_scaler, create_model
from process_data import (process_data_GK, process_data_DC, process_data_FB,
                          process_data_MD, process_data_WG, process_data_FW)

app = Flask(__name__)

attribute_names_GK = [
    'PSxG-GA', 'Goals Against', 'Save Percentage', 'PSxG/SoT',
    'Save% (Penalty Kicks)', 'Clean Sheet Percentage', 'Touches',
    'Launch %', 'Goal Kicks', 'Avg. Length of Goal Kicks',
    'Crosses Stopped %', 'Def. Actions Outside Pen. Area'
]

attribute_names_DC = ['Tackles', 'Interceptions', 'Blocks', 'Clearances', 'Aerials won', 'Pass Completion %']

attribute_names_FB = ['npxG + xAG', 'Tackles', 'Interceptions', 'Blocks', 'Clearances', 'Aerials won',
                      'Pass Completion %']

attribute_names_MD = ['Tackles', 'Interceptions', 'npxG + xAG', 'Shot-Creating Actions', 'Passes Attempted',
                      'Pass Completion %', 'Progressive Passes', 'Progressive Carries',
                      'Successful Take-Ons']

attribute_names_WG = ['Non-Penalty Goals', 'Non-Penalty xG', 'Shots Total', 'Assists',
                      'xAG', 'npxG + xAG', 'Shot-Creating Actions', 'Passes Attempted',
                      'Pass Completion %', 'Progressive Passes', 'Progressive Carries',
                      'Successful Take-Ons']

attribute_names_FW = ['Non-Penalty Goals', 'Non-Penalty xG', 'Shots Total', 'Assists', 'xAG', 'npxG + xAG',
                      'Shot-Creating Actions']

GK_model = load_model('GK_model.pth', attribute_names_GK)
DC_model = load_model('DC_model.pth', attribute_names_DC)
FB_model = load_model('FB_model.pth', attribute_names_FB)
MD_model = load_model('MD_model.pth', attribute_names_MD)
WG_model = load_model('WG_model.pth', attribute_names_WG)
FW_model = load_model('FW_model.pth', attribute_names_FW)

scaler_GK = load_scaler()
scaler_DC = load_scaler()
scaler_FB = load_scaler()
scaler_MD = load_scaler()
scaler_WG = load_scaler()
scaler_FW = load_scaler()


def predict(model, scaler, data, names, top_n):
    data_scaled = scaler.transform(data)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        predictions = model(data_tensor).numpy()
    best_indices = np.argsort(predictions, axis=0)[-top_n:][::-1]  # Отримати індекси найкращих гравців
    return [names[idx][0] for idx in best_indices]  # Вивести лише імена гравців


@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        selected_tactics = request.form.get('tactics')

        if all(key in request.files for key in
               ['goalkeeper_dataset', 'defender_dataset', 'fullback_dataset', 'midfielder_dataset', 'winger_dataset',
                'forward_dataset']):
            goalkeeper_data, _, goalkeeper_names = process_data_GK(request.files['goalkeeper_dataset'])
            defender_data, _, defender_names = process_data_DC(request.files['defender_dataset'])
            fullback_data, _, fullback_names = process_data_FB(request.files['fullback_dataset'])
            midfielder_data, _, midfielder_names = process_data_MD(request.files['midfielder_dataset'])
            winger_data, _, winger_names = process_data_WG(request.files['winger_dataset'])
            forward_data, _, forward_names = process_data_FW(request.files['forward_dataset'])

            scaler_GK.fit(goalkeeper_data)
            scaler_DC.fit(defender_data)
            scaler_FB.fit(fullback_data)
            scaler_MD.fit(midfielder_data)
            scaler_WG.fit(winger_data)
            scaler_FW.fit(forward_data)
            best_winger = []
            if selected_tactics == '4-3-3':
                best_goalkeeper = predict(GK_model, scaler_GK, goalkeeper_data, goalkeeper_names, 1)
                best_defender = predict(DC_model, scaler_DC, defender_data, defender_names, 2)
                best_fullback = predict(FB_model, scaler_FB, fullback_data, fullback_names, 2)
                best_midfielder = predict(MD_model, scaler_MD, midfielder_data, midfielder_names, 3)
                best_winger = predict(WG_model, scaler_WG, winger_data, winger_names, 2)
                best_forward = predict(FW_model, scaler_FW, forward_data, forward_names, 1)
            elif selected_tactics == '4-4-2':
                best_goalkeeper = predict(GK_model, scaler_GK, goalkeeper_data, goalkeeper_names, 1)
                best_defender = predict(DC_model, scaler_DC, defender_data, defender_names, 2)
                best_fullback = predict(FB_model, scaler_FB, fullback_data, fullback_names, 2)
                best_midfielder = predict(MD_model, scaler_MD, midfielder_data, midfielder_names, 2)
                best_winger = predict(WG_model, scaler_WG, winger_data, winger_names, 2)
                best_forward = predict(FW_model, scaler_FW, forward_data, forward_names, 2)
            elif selected_tactics == '5-3-2':
                best_goalkeeper = predict(GK_model, scaler_GK, goalkeeper_data, goalkeeper_names, 1)
                best_defender = predict(DC_model, scaler_DC, defender_data, defender_names, 3)
                best_fullback = predict(FB_model, scaler_FB, fullback_data, fullback_names, 2)
                best_midfielder = predict(MD_model, scaler_MD, midfielder_data, midfielder_names, 3)
                best_forward = predict(FW_model, scaler_FW, forward_data, forward_names, 2)
            else:
                return "Invalid tactics provided", 400

            return render_template('result.html',
                                   best_goalkeeper=best_goalkeeper,
                                   best_defender=best_defender,
                                   best_fullback=best_fullback,
                                   best_midfielder=best_midfielder,
                                   best_winger=best_winger,
                                   best_forward=best_forward)
        else:
            return "Missing data files in the request", 400
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict_all_positions():
    position_mapping = {
        'goalkeeper': (GK_model, scaler_GK, attribute_names_GK),
        'defender': (DC_model, scaler_DC, attribute_names_DC),
        'fullback': (FB_model, scaler_FB, attribute_names_FB),
        'midfielder': (MD_model, scaler_MD, attribute_names_MD),
        'winger': (WG_model, scaler_WG, attribute_names_WG),
        'forward': (FW_model, scaler_FW, attribute_names_FW)
    }

    position = request.form.get('position')

    if position not in position_mapping:
        return "Invalid position provided", 400

    position_model, position_scaler, attribute_names = position_mapping[position]

    dataset_key = f'{position}_dataset'
    if dataset_key not in request.files:
        return f"Missing {position} dataset in the request", 400

    data = request.files[dataset_key]
    best_player = predict(position_model, position_scaler, data, attribute_names)
    return jsonify({'best_player': best_player})


if __name__ == '__main__':
    app.run(debug=True)
