import os
import math
import numpy as np

# Ensure PyTorch backend is selected for Keras 3
os.environ["KERAS_BACKEND"] = "torch"
import keras
from .preprocessor import AISPreprocessor

class LSTMPredictor:
    """
    Manages loading and inference for the trained vessel_predictor.keras LSTM model.
    Calculates trajectory prediction, displacement deviation, and kinetic anomaly metrics.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default to models/vessel_predictor.keras relative to repo root
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            model_path = os.path.join(base_dir, "models", "vessel_predictor.keras")

        self.model_path = model_path
        self.preprocessor = AISPreprocessor()
        self.model = None
        self.loaded = False
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                self.loaded = True
                print(f"[LSTM Engine] Successfully synchronized weights from {self.model_path}")
            else:
                print(f"[LSTM Engine] Warning: Model file not found at {self.model_path}")
        except Exception as e:
            print(f"[LSTM Engine] Error loading model: {e}")
            self.loaded = False

    def predict_next_position(self, history: list):
        """
        Runs the 10-step history through the LSTM.
        Returns (predicted_lat, predicted_lon).
        """
        if not self.loaded or not self.model:
            # Fallback dead reckoning if model offline
            last = history[-1] if history else [10.0, 75.0, 10.0, 0.0]
            lat, lon, sog, cog = last[0], last[1], last[2], last[3]
            dt = 0.02
            pred_lat = lat + (sog * dt) * math.cos(math.radians(cog))
            pred_lon = lon + (sog * dt) * math.sin(math.radians(cog))
            return round(pred_lat, 4), round(pred_lon, 4)

        inp = self.preprocessor.process_sequence(history)
        raw_pred = self.model.predict(inp, verbose=0)
        norm_lat, norm_lon = float(raw_pred[0][0]), float(raw_pred[0][1])
        pred_lat, pred_lon = self.preprocessor.denormalize_lat_lon(norm_lat, norm_lon)
        return round(pred_lat, 4), round(pred_lon, 4)

    @staticmethod
    def haversine_km(lat1, lon1, lat2, lon2):
        """Great-circle distance between two points on a sphere in kilometers."""
        r = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    def evaluate_vessel_risk(self, vessel: dict):
        """
        Calculates mathematical IUU risk index using:
        1. Model trajectory deviation: predicted vs actual displacement
        2. Kinetic anomalies: loitering, speed drops, erratic heading variance
        3. Identity & flag irregularities
        4. Protected maritime zones
        """
        history = vessel.get("history", [])
        lat = vessel.get("lat", 0.0)
        lon = vessel.get("lon", 0.0)
        sog = vessel.get("sog", 0.0)
        vtype = vessel.get("vtype", "Cargo")
        flag = vessel.get("flag", "Unknown")

        pred_lat, pred_lon = self.predict_next_position(history)
        disp_error_km = self.haversine_km(lat, lon, pred_lat, pred_lon)

        score = 0.0
        reasons = []

        # 1. Trajectory Anomaly Score (0 - 35 pts)
        if disp_error_km > 15.0:
            score += min(35.0, (disp_error_km - 15.0) * 1.5 + 15.0)
            reasons.append(f"Kinetic Course Deviation ({disp_error_km:.1f} km off predicted vector)")

        # 2. Loitering / Extraction Pattern (0 - 25 pts)
        if vtype in ["Fishing", "Trawler"] and sog < 3.0:
            score += 25.0
            reasons.append("Active Extraction Signature: SOG < 3.0 kn in open corridor")
        elif vtype == "Tanker" and sog < 2.0:
            score += 20.0
            reasons.append("Ship-to-Ship (STS) Loitering Risk: Drift velocity observed")

        # 3. Heading Instability / Zig-zagging (0 - 20 pts)
        cog_drift = abs(vessel.get("cog_drift", 0.0))
        if cog_drift > 2.5:
            score += min(20.0, cog_drift * 4.0)
            reasons.append("Erratic Yaw Fluctuations: Evasive maneuvering signature")

        # 4. Identity / Flag State Penalty (0 - 15 pts)
        if flag == "Unknown":
            score += 15.0
            reasons.append("Identity Anomaly: Non-registered or masked flag state")

        # 5. Marine Protected Zone Proximity (0 - 20 pts)
        # Coordinates around Lakshadweep / Nicobar MPA coordinates
        if (8.0 < lat < 12.5 and 71.5 < lon < 74.0) or (6.5 < lat < 14.0 and 92.0 < lon < 94.0):
            score += 20.0
            reasons.append("Protected Marine Area (MPA) Sanctuary Infringement")

        final_risk = float(np.clip(score, 5.0, 98.5))
        level = "HIGH" if final_risk >= 70.0 else ("MEDIUM" if final_risk >= 40.0 else "LOW")

        return {
            "predicted_lat": pred_lat,
            "predicted_lon": pred_lon,
            "deviation_km": round(disp_error_km, 2),
            "risk_score": round(final_risk, 1),
            "risk_level": level,
            "flags": reasons if reasons else ["NOMINAL_TRANSIT"]
        }
