import ast
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


models_dir = "C:\\Users\\Dima\\Desktop\\Навчання\\Дипломна"

def load_scaler():
    return StandardScaler()


def create_model(input_size):
    class Model(nn.Module):
        def __init__(self, input_size):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 128)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(128, 64)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(64, 1)
            self.regularization = nn.Linear(input_size, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.fc4(x)

            reg_loss = torch.norm(self.regularization.weight, 2)
            return x + reg_loss

    return Model(input_size)


def load_model(model_name, attribute_names):
    model_path = os.path.join(models_dir, model_name)
    if os.path.exists(model_path):
        model = create_model(input_size=len(attribute_names))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Файл моделі не знайдено: {model_name}")
