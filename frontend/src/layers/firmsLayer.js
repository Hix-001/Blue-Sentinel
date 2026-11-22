import * as Cesium from 'cesium';
import { requestFrame } from '../core/renderGovernor.js';

/**
 * NASA FIRMS Thermal Hotspot Layer.
 * Renders thermal infrared detections on the marine surface
 * to highlight gas flaring, engine thermal blooms, and night activity.
 */

export class FirmsThermalLayer {
  constructor(viewer) {
    this.viewer = viewer;
    this.points = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
    this.visible = true;
    this._generateSampleHotspots();
  }

  _generateSampleHotspots() {
    // Concentrated thermal anomalies around high-density offshore rigs & sea corridors
    const hotspots = [
      { lat: 19.35, lon: 71.45, brightness: 345.2, name: "Bombay High Offshore Flare" },
      { lat: 18.90, lon: 72.10, brightness: 330.1, name: "Offshore Gas Flaring Complex" },
      { lat: 12.15, lon: 79.95, brightness: 320.5, name: "Coromandel Thermal Extraction" },
      { lat: 8.45, lon: 77.20, brightness: 338.0, name: "Cape Comorin Thermal Anomaly" },
      { lat: 16.20, lon: 82.40, brightness: 350.4, name: "Krishna-Godavari Deepwater Rig" }
    ];

    for (const spot of hotspots) {
      this.points.add({
        position: Cesium.Cartesian3.fromDegrees(spot.lon, spot.lat, 50.0),
        pixelSize: 16,
        color: Cesium.Color.fromCssColorString('#ff4500').withAlpha(0.75),
        outlineColor: Cesium.Color.YELLOW,
        outlineWidth: 2
      });
    }
    requestFrame();
  }

  setVisible(visible) {
    this.visible = visible;
    this.points.show = visible;
    requestFrame();
  }
}
