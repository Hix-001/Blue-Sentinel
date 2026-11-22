import time
import random
import math
import numpy as np
from ..ml.predictor import LSTMPredictor

NAMES = {
    "Cargo":   ["MV Sindhu Star", "MV Bay Trader", "MV Bengal Pride", "MV Coromandel", "MV Vizag Express",
                "MV Eastern Promise", "MV Kochi Venture", "MV Malabar Queen", "MV Chennai Spirit", "MV Paradip Bulk"],
    "Tanker":  ["MT Arabian Dawn", "MT Gulf Stream", "MT Chennai Crude", "MT Visakha", "MT Andaman Maru",
                "MT Lakshadweep", "MT Palk Strait", "MT Oman Sky", "MT Hormuz", "MT Karwar"],
    "Fishing": ["FV Sagar Devi", "FV Deep Blue", "FV Ocean Catch", "FV Matsya", "FV Trivandrum",
                "FV Bay Fisher", "FV Calicut Net", "FV Ratnagiri", "FV Latur Sea", "FV Puri Pearl"],
    "Security":["INS Talwar", "INS Vikrant", "INS Kolkata", "CG Sarathi", "INS Shivalik",
                "INS Sahyadri", "INS Kamorta", "INS Kadmatt", "INS Kiltan", "INS Kavaratti"],
    "Research":["RV Sagar Nidhi", "RV Sindhu Sadhana", "RV Sagar Kanya", "RV Samudra Ratna",
                "RV Samudra Shakti", "RV Samudra Sewak", "RV Sagar Anveshika", "RV Deep Nidhi"],
}
BASE_SOG = {"Cargo": 14.0, "Tanker": 12.0, "Fishing": 7.5, "Security": 22.0, "Research": 10.0}

class FleetManager:
    """
    Simulates and maintains a live mesh of vessels across the Indo-Pacific / Indian Ocean.
    Calculates step kinematics, manages historical trail vectors, and computes real-time LSTM risk.
    """
    def __init__(self, count: int = 60):
        self.predictor = LSTMPredictor()
        self.vessels = self._init_fleet(count)
        self.tick = 0

    def _init_fleet(self, count: int):
        rng = random.Random(42)
        vtypes = ["Cargo", "Tanker", "Fishing", "Security", "Research"]
        weights = [30, 20, 32, 8, 10]
        flags = ["India", "Singapore", "China", "Panama", "Norway", "Sri Lanka", "Liberia", "Unknown"]
        
        fleet = {}
        for i in range(count):
            vt = rng.choices(vtypes, weights=weights, k=1)[0]
            name = f"{rng.choice(NAMES[vt])}-{rng.randint(1, 99):02d}"
            lat = rng.uniform(4.0, 22.0)
            lon = rng.uniform(60.0, 95.0)
            base_sog = BASE_SOG[vt]
            sog = base_sog + rng.uniform(-1.5, 1.5)
            cog = rng.uniform(0.0, 360.0)
            mmsi = int(rng.randint(211000000, 775000000))
            
            # Initial history
            history = []
            cur_lat, cur_lon = lat, lon
            for _ in range(10):
                cur_lat += rng.uniform(-0.015, 0.015)
                cur_lon += rng.uniform(-0.015, 0.015)
                history.append([round(cur_lat, 4), round(cur_lon, 4), round(sog, 1), round(cog, 1)])

            fleet[mmsi] = {
                "mmsi": mmsi,
                "name": name,
                "vtype": vt,
                "flag": rng.choice(flags),
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "sog": round(sog, 1),
                "cog": round(cog, 1),
                "cog_drift": round(rng.uniform(-1.0, 1.0), 2),
                "base_sog": base_sog,
                "history": history,
                "risk_score": 15.0,
                "risk_level": "LOW",
                "flags": ["NOMINAL_TRANSIT"],
                "predicted_lat": round(lat, 4),
                "predicted_lon": round(lon, 4)
            }
        return fleet

    def step(self):
        """Advances fleet kinematics by one simulation tick and evaluates AI risk."""
        self.tick += 1
        for mmsi, v in self.vessels.items():
            # Kinematic speed oscillation
            v["sog"] += random.uniform(-0.8, 0.8)
            v["sog"] = float(np.clip(v["sog"], v["base_sog"] * 0.3, v["base_sog"] * 1.5))

            # Heading drift
            v["cog_drift"] += random.uniform(-0.3, 0.3)
            v["cog_drift"] = float(np.clip(v["cog_drift"], -4.0, 4.0))
            v["cog"] = (v["cog"] + v["cog_drift"] + random.uniform(-2.0, 2.0)) % 360.0

            # Coordinate step (1 kn ~ 0.0003 deg/sec scale for visible animation)
            step_mag = v["sog"] * 0.00035
            v["lat"] += step_mag * math.cos(math.radians(v["cog"]))
            v["lon"] += step_mag * math.sin(math.radians(v["cog"]))

            # Soft boundary bouncing for Indian Ocean bounds (Lat 1-25 N, Lon 56-100 E)
            if not (2.0 < v["lat"] < 25.0):
                v["cog"] = (v["cog"] + 180.0) % 360.0
                v["lat"] = float(np.clip(v["lat"], 2.5, 24.5))
            if not (56.0 < v["lon"] < 98.0):
                v["cog"] = (v["cog"] + 180.0) % 360.0
                v["lon"] = float(np.clip(v["lon"], 57.0, 97.0))

            v["lat"] = round(v["lat"], 4)
            v["lon"] = round(v["lon"], 4)
            v["sog"] = round(v["sog"], 1)
            v["cog"] = round(v["cog"], 1)

            # Update historical trail
            v["history"].append([v["lat"], v["lon"], v["sog"], v["cog"]])
            if len(v["history"]) > 20:
                v["history"].pop(0)

            # Evaluate with LSTM Model
            analysis = self.predictor.evaluate_vessel_risk(v)
            v["predicted_lat"] = analysis["predicted_lat"]
            v["predicted_lon"] = analysis["predicted_lon"]
            v["risk_score"] = analysis["risk_score"]
            v["risk_level"] = analysis["risk_level"]
            v["flags"] = analysis["flags"]

        return list(self.vessels.values())

    def get_fleet_snapshot(self):
        return list(self.vessels.values())
