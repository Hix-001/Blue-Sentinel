import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .telemetry.ais_service import FleetManager
from .telemetry.weather import fetch_marine_weather

fleet_manager = FleetManager(count=75)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start live AISStream ingestion task in background if configured in .env
    bg_task = asyncio.create_task(fleet_manager.start_live_ingestion())
    yield
    # Shutdown: Cancel background ingestion
    bg_task.cancel()

app = FastAPI(
    title="Blue Sentinel Maritime Intelligence Engine",
    description="Decoupled AI/ML C4ISR Telemetry Service with LSTM Anomaly Detection",
    version="5.0",
    lifespan=lifespan
)

# Enable CORS for the Vite/Cesium frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {
        "status": "ONLINE",
        "node_id": "INDO-PAC-DL04",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "neural_core": "SYNCHRONIZED" if fleet_manager.predictor.loaded else "STANDBY",
        "live_aisstream": "ACTIVE" if fleet_manager.live_connected else "SIMULATION_FALLBACK",
        "active_assets": len(fleet_manager.vessels)
    }

@app.get("/api/fleet")
def get_fleet():
    """Returns current state snapshot of all tracked maritime assets."""
    return {
        "count": len(fleet_manager.vessels),
        "tick": fleet_manager.tick,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "live_uplink": fleet_manager.live_connected,
        "vessels": fleet_manager.get_fleet_snapshot()
    }

@app.get("/api/vessel/{mmsi}")
def get_vessel_detail(mmsi: int):
    """Returns complete state, historical trail, and AI explainability flags for one vessel."""
    vessel = fleet_manager.vessels.get(mmsi)
    if not vessel:
        return {"error": "Vessel not found", "mmsi": mmsi}
    return vessel

@app.get("/api/weather")
async def get_weather(lat: float = 14.0, lon: float = 78.0):
    return await fetch_marine_weather(lat, lon)

@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    """
    High-frequency WebSocket channel pushing fleet updates,
    kinetic positions, and LSTM risk evaluations every 2 seconds.
    """
    await websocket.accept()
    print("[Telemetry Stream] Frontend operator connected.")
    try:
        while True:
            # Advance simulation step and evaluate LSTM predictions
            vessels = fleet_manager.step()
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tick": fleet_manager.tick,
                "live_connected": fleet_manager.live_connected,
                "stats": {
                    "total": len(vessels),
                    "high_risk": sum(1 for v in vessels if v["risk_level"] == "HIGH"),
                    "medium_risk": sum(1 for v in vessels if v["risk_level"] == "MEDIUM"),
                    "nominal": sum(1 for v in vessels if v["risk_level"] == "LOW"),
                },
                "vessels": vessels
            }
            await websocket.send_json(payload)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        print("[Telemetry Stream] Frontend operator disconnected.")
    except Exception as e:
        print(f"[Telemetry Stream] Error: {e}")
