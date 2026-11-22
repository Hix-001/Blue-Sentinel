import os
import json
import time
import random
import math
import asyncio
import websockets
import numpy as np
from dotenv import load_dotenv
from ..ml.predictor import LSTMPredictor

# Load environment variables from .env or .env.txt (Windows Notepad fallback)
load_dotenv(".env")
if not os.getenv("AISSTREAM_API_KEY"):
    load_dotenv(".env.txt")

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
    Ingests live AISStream.io telemetry when AISSTREAM_API_KEY is provided in .env,
    otherwise runs high-fidelity simulated kinematic walks.
    """
    def __init__(self, count: int = 60):
        self.predictor = LSTMPredictor()
        self.vessels = self._init_fleet(count)
        self.tick = 0
        self.api_key = os.getenv("AISSTREAM_API_KEY", "").strip()
        self.live_connected = False
        self._bg_task = None

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
                "predicted_lon": round(lon, 4),
                "source": "SIMULATION"
            }
        return fleet

    async def start_live_ingestion(self):
        """Asynchronous worker that ingests live AIS messages from AISStream.io if API key exists."""
        if not self.api_key or self.api_key == "your_aisstream_api_key_here":
            print("[AIS Ingestion] No live AISStream API key configured in .env. Running in simulation mode.")
            return

        subscribe_message = {
            "APIKey": self.api_key,
            "BoundingBoxes": [[[0.0, 50.0], [30.0, 105.0]]],  # Indo-Pacific bounding box
            "FilterMessageTypes": ["PositionReport", "StandardClassBPositionReport"]
        }

        url = "wss://stream.aisstream.io/v0/stream"
        print("[AIS Ingestion] Attempting secure uplink to wss://stream.aisstream.io...")

        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps(subscribe_message))
                    self.live_connected = True
                    print("[AIS Ingestion] LIVE AISStream uplink established. Ingesting commercial maritime beacons.")

                    async for message in ws:
                        record = json.loads(message)
                        msg_type = record.get("MessageType")
                        meta = record.get("MetaData", {})
                        body = record.get("Message", {}).get(msg_type, {})

                        mmsi = meta.get("MMSI")
                        lat = meta.get("latitude")
                        lon = meta.get("longitude")
                        name = (meta.get("ShipName") or f"VESSEL-{mmsi}").strip()

                        if mmsi and lat is not None and lon is not None:
                            sog = float(body.get("Sog", 10.0))
                            cog = float(body.get("Cog", 0.0))
                            self._update_live_vessel(mmsi, name, lat, lon, sog, cog)

            except Exception as e:
                self.live_connected = False
                print(f"[AIS Ingestion] AISStream connection transient error: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10.0)

    def _update_live_vessel(self, mmsi: int, name: str, lat: float, lon: float, sog: float, cog: float):
        entry = self.vessels.get(mmsi)
        if not entry:
            entry = {
                "mmsi": mmsi,
                "name": name,
                "vtype": "Cargo",
                "flag": "Commercial",
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "sog": round(sog, 1),
                "cog": round(cog, 1),
                "cog_drift": 0.0,
                "base_sog": 12.0,
                "history": [[round(lat, 4), round(lon, 4), round(sog, 1), round(cog, 1)]],
                "risk_score": 10.0,
                "risk_level": "LOW",
                "flags": ["LIVE_BROADCAST"],
                "predicted_lat": round(lat, 4),
                "predicted_lon": round(lon, 4),
                "source": "LIVE_AISSTREAM"
            }
            self.vessels[mmsi] = entry
        else:
            entry["lat"] = round(lat, 4)
            entry["lon"] = round(lon, 4)
            entry["sog"] = round(sog, 1)
            entry["cog"] = round(cog, 1)
            entry["history"].append([entry["lat"], entry["lon"], entry["sog"], entry["cog"]])
            if len(entry["history"]) > 20:
                entry["history"].pop(0)

            # Evaluate through the LSTM engine
            analysis = self.predictor.evaluate_vessel_risk(entry)
            entry["predicted_lat"] = analysis["predicted_lat"]
            entry["predicted_lon"] = analysis["predicted_lon"]
            entry["risk_score"] = analysis["risk_score"]
            entry["risk_level"] = analysis["risk_level"]
            entry["flags"] = analysis["flags"]

    def step(self):
        """Advances fleet kinematics by one simulation tick and evaluates AI risk."""
        self.tick += 1
        for mmsi, v in list(self.vessels.items()):
            # Only apply kinematic random walk to simulated vessels
            if v.get("source") != "LIVE_AISSTREAM":
                v["sog"] += random.uniform(-0.8, 0.8)
                v["sog"] = float(np.clip(v["sog"], v["base_sog"] * 0.3, v["base_sog"] * 1.5))

                v["cog_drift"] += random.uniform(-0.3, 0.3)
                v["cog_drift"] = float(np.clip(v["cog_drift"], -4.0, 4.0))
                v["cog"] = (v["cog"] + v["cog_drift"] + random.uniform(-2.0, 2.0)) % 360.0

                step_mag = v["sog"] * 0.00035
                v["lat"] += step_mag * math.cos(math.radians(v["cog"]))
                v["lon"] += step_mag * math.sin(math.radians(v["cog"]))

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

                v["history"].append([v["lat"], v["lon"], v["sog"], v["cog"]])
                if len(v["history"]) > 20:
                    v["history"].pop(0)

                analysis = self.predictor.evaluate_vessel_risk(v)
                v["predicted_lat"] = analysis["predicted_lat"]
                v["predicted_lon"] = analysis["predicted_lon"]
                v["risk_score"] = analysis["risk_score"]
                v["risk_level"] = analysis["risk_level"]
                v["flags"] = analysis["flags"]

        return list(self.vessels.values())

    def get_fleet_snapshot(self):
        return list(self.vessels.values())
