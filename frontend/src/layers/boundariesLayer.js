import * as Cesium from 'cesium';
import { requestFrame } from '../core/renderGovernor.js';

/**
 * Natural Earth Marine Boundaries Layer.
 * Defines exclusive economic zones (EEZs), seas, and gulfs
 * for detecting unauthorized territorial incursions.
 */

export class MarineBoundariesLayer {
  constructor(viewer) {
    this.viewer = viewer;
    this.dataSource = null;
    this.visible = true;
    this._load();
  }

  async _load() {
    try {
      const response = await fetch('/data/natural_earth/marine.json');
      const geojson = await response.json();

      this.dataSource = await Cesium.GeoJsonDataSource.load(geojson, {
        stroke: Cesium.Color.fromCssColorString('#7b5fff').withAlpha(0.5),
        strokeWidth: 1.5,
        fill: Cesium.Color.fromCssColorString('#7b5fff').withAlpha(0.04),
        clampToGround: true
      });

      this.viewer.dataSources.add(this.dataSource);
      console.log("[Marine Boundaries] Loaded territorial polygons.");
      requestFrame();
    } catch (err) {
      console.warn("[Marine Boundaries] Unable to load marine geojson:", err);
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
