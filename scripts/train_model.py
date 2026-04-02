#train model import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class AISPreprocessor:
    def __init__(self):
        self.lat_min, self.lat_max = -90.0, 90.0
        self.lon_min, self.lon_max = -180.0, 180.0
        self.sog_min, self.sog_max = 0.0, 30.0
        
    def normalize_lat_lon(self, lat, lon):
        norm_lat = (lat - self.lat_min) / (self.lat_max - self.lat_min)
        norm_lon = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        return norm_lat, norm_lon
        
    def denormalize_lat_lon(self, norm_lat, norm_lon):
        lat = norm_lat * (self.lat_max - self.lat_min) + self.lat_min
        lon = norm_lon * (self.lon_max - self.lon_min) + self.lon_min
        return lat, lon

    def normalize_sog(self, sog):
        return (sog - self.sog_min) / (self.sog_max - self.sog_min)

    def encode_cog(self, cog):
        rad = np.radians(cog)
        return np.sin(rad), np.cos(rad)

    def process_point(self, lat, lon, sog, cog):
        norm_lat, norm_lon = self.normalize_lat_lon(lat, lon)
        norm_sog = self.normalize_sog(sog)
        sin_cog, cos_cog = self.encode_cog(cog)
        return [norm_lat, norm_lon, norm_sog, sin_cog, cos_cog]

def generate_synthetic_ais_sequences(num_sequences=1000, time_steps=10):
    preprocessor = AISPreprocessor()
    X, y = [], []
    
    for _ in range(num_sequences):
        lat = np.random.uniform(5.0, 20.0)
        lon = np.random.uniform(70.0, 85.0)
        sog = np.random.uniform(2.0, 15.0)
        cog = np.random.uniform(0, 360)
        
        sequence = []
        for _ in range(time_steps):
            sequence.append(preprocessor.process_point(lat, lon, sog, cog))
            lat += (sog * 0.005) * np.cos(np.radians(cog))
            lon += (sog * 0.005) * np.sin(np.radians(cog))
            cog = (cog + np.random.uniform(-5, 5)) % 360
            
        X.append(sequence)
        target_lat, target_lon = preprocessor.normalize_lat_lon(lat, lon)
        y.append([target_lat, target_lon])
        
    return np.array(X), np.array(y)

def build_lstm_model(time_steps, features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, features)),
        Dropout(0.2),
        LSTM(32),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    TIME_STEPS = 10
    FEATURES = 5
    
    X_train, y_train = generate_synthetic_ais_sequences(num_sequences=2500, time_steps=TIME_STEPS)
    model = build_lstm_model(TIME_STEPS, FEATURES)
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)
    model.save("../models/vessel_predictor.keras")