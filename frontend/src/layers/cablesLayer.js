import * as Cesium from 'cesium';
import { requestFrame } from '../core/renderGovernor.js';

/**
 * Submarine Cable Infrastructure Layer for Blue Sentinel.
 * Visualizes global subsea communications cables to detect suspicious
 * loitering or anchoring over critical maritime infrastructure.
 */

export class SubmarineCablesLayer {
  constructor(viewer) {
    this.viewer = viewer;
    this.dataSource = null;
    this.visible = true;
    this._load();
  }

  async _load() {
    try {
      const response = await fetch('/data/cables/cable-geo.json');
      const geojson = await response.json();
      
      this.dataSource = await Cesium.GeoJsonDataSource.load(geojson, {
        stroke: Cesium.Color.fromCssColorString('#00ffbb').withAlpha(0.65),
        strokeWidth: 2,
        clampToGround: true
      });

      this.viewer.dataSources.add(this.dataSource);
      console.log("[Submarine Cables] Loaded undersea infrastructure telemetry.");
      requestFrame();
    } catch (err) {
      console.warn("[Submarine Cables] Unable to load cables geojson:", err);
    }
  }

  setVisible(visible) {
    this.visible = visible;
    if (this.dataSource) {
      this.dataSource.show = visible;
      requestFrame();
    }
  }
}
