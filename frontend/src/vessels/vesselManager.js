import * as Cesium from 'cesium';
import { requestFrame } from '../core/renderGovernor.js';

/**
 * Vessel Fleet Manager for Blue Sentinel.
 * Connects to FastAPI backend WebSocket, renders vessels as rotating boat sprites,
 * kinetic wake trails, target locks, and focus de-emphasis.
 */

export class VesselManager {
  constructor(viewer, onSelectVessel) {
    this.viewer = viewer;
    this.onSelectVessel = onSelectVessel;
    this.vessels = new Map();
    this.billboards = viewer.scene.primitives.add(new Cesium.BillboardCollection());
    this.trails = viewer.scene.primitives.add(new Cesium.PolylineCollection());
    this.selectedMmsi = null;
    this.showTrails = true;
    this.showVessels = true;

    this._setupPicking();
    this._connectWebSocket();
  }

  _connectWebSocket() {
    const defaultWs = window.location.protocol === 'https:'
      ? `wss://${window.location.host}/ws/telemetry`
      : `ws://${window.location.hostname}:8000/ws/telemetry`;
    const wsUrl = import.meta.env.VITE_WS_URL || defaultWs;
    console.log(`[AIS Stream] Connecting to WebSocket: ${wsUrl}`);
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log("[AIS Stream] Connected to FastAPI LSTM Telemetry Service.");
      const ticker = document.getElementById("ticker-text");
      if (ticker) ticker.innerText = "UPLINK SYNCHRONIZED — INGESTING LIVE LSTM ANOMALY MESH";
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.updateFleet(data.vessels, data.stats);
      } catch (err) {
        console.error("Telemetry parse error:", err);
      }
    };

    socket.onclose = () => {
      console.warn("[AIS Stream] WebSocket disconnected. Falling back to HTTP polling in 3s...");
      setTimeout(() => this._pollHttp(), 3000);
    };
  }

  async _pollHttp() {
    try {
      const baseUrl = import.meta.env.VITE_API_URL || '';
      const res = await fetch(`${baseUrl}/api/fleet`);
      const data = await res.json();
      if (data.vessels) {
        this.updateFleet(data.vessels);
      }
    } catch (e) {}
    setTimeout(() => this._pollHttp(), 2500);
  }

  updateFleet(vesselList, stats = null) {
    if (!vesselList) return;

    if (stats) {
      const fleetEl = document.getElementById("fleet-count");
      const highEl = document.getElementById("high-risk-count");
      if (fleetEl) fleetEl.innerText = `${stats.total} ASSETS`;
      if (highEl) highEl.innerText = `${stats.high_risk} CRITICAL`;
    }

    for (const v of vesselList) {
      let entry = this.vessels.get(v.mmsi);

      // Height clamping above geoid ocean surface
      const pos = Cesium.Cartesian3.fromDegrees(v.lon, v.lat, 25.0);
      const isHigh = v.risk_level === "HIGH";
      const isMed = v.risk_level === "MEDIUM";

      let color = Cesium.Color.CYAN;
      let width = 22;
      let height = 22;

      if (isHigh) {
        color = Cesium.Color.fromCssColorString('#ff2d55');
        width = 30;
        height = 30;
      } else if (isMed) {
        color = Cesium.Color.fromCssColorString('#ffb340');
        width = 26;
        height = 26;
      }

      // Focus De-emphasis: dim other vessels if one is selected
      if (this.selectedMmsi) {
        if (v.mmsi === this.selectedMmsi) {
          color = Cesium.Color.LIME;
          width = 36;
          height = 36;
        } else {
          color = color.withAlpha(0.22);
        }
      }

      // Heading rotation (Cesium billboard rotation is counter-clockwise, COG is clockwise from North)
      const rotationRad = -Cesium.Math.toRadians(v.cog || 0.0);

      if (!entry) {
        // Create tactical boat billboard sprite
        const billboard = this.billboards.add({
          position: pos,
          image: '/assets/boat.png',
          width: width,
          height: height,
          rotation: rotationRad,
          color: color,
          scaleByDistance: new Cesium.NearFarScalar(5.0e3, 1.3, 8.0e6, 0.45),
          id: v.mmsi
        });

        // Create kinetic trail
        const trailPositions = (v.history || []).map(h => Cesium.Cartesian3.fromDegrees(h[1], h[0], 10.0));
        const trail = this.trails.add({
          positions: trailPositions,
          width: isHigh ? 2.5 : 1.5,
          material: Cesium.Material.fromType('Color', {
            color: color.withAlpha(0.4)
          })
        });

        entry = { billboard, trail, data: v };
        this.vessels.set(v.mmsi, entry);
      } else {
        entry.data = v;
        entry.billboard.position = pos;
        entry.billboard.color = color;
        entry.billboard.width = width;
        entry.billboard.height = height;
        entry.billboard.rotation = rotationRad;

        // Update trail positions
        if (v.history && v.history.length >= 2) {
          const trailPositions = v.history.map(h => Cesium.Cartesian3.fromDegrees(h[1], h[0], 10.0));
          entry.trail.positions = trailPositions;
        }
      }

      // If currently inspected, refresh drawer
      if (this.selectedMmsi === v.mmsi && this.onSelectVessel) {
        this.onSelectVessel(v);
      }
    }

    requestFrame();
  }

  _setupPicking() {
    const handler = new Cesium.ScreenSpaceEventHandler(this.viewer.scene.canvas);
    handler.setInputAction((movement) => {
      const picked = this.viewer.scene.pick(movement.position);
      if (Cesium.defined(picked) && picked.id) {
        const mmsi = picked.id;
        const vessel = this.vessels.get(mmsi);
        if (vessel) {
          this.selectVessel(mmsi);
        }
      }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
  }

  selectVessel(mmsi) {
    this.selectedMmsi = mmsi;
    const vessel = this.vessels.get(mmsi);
    if (!vessel) return;

    const v = vessel.data;

    // Cinematic Camera Target Lock & Follow
    this.viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(v.lon, v.lat, 45000.0),
      orientation: {
        heading: Cesium.Math.toRadians(v.cog),
        pitch: Cesium.Math.toRadians(-45.0),
        roll: 0.0
      },
      duration: 1.8
    });

    if (this.onSelectVessel) {
      this.onSelectVessel(v);
    }
    requestFrame();
  }

  clearSelection() {
    this.selectedMmsi = null;
    requestFrame();
  }

  setTrailsVisible(visible) {
    this.showTrails = visible;
    this.trails.show = visible;
    requestFrame();
  }

  setVesselsVisible(visible) {
    this.showVessels = visible;
    this.billboards.show = visible;
    this.trails.show = visible && this.showTrails;
    requestFrame();
  }
}
