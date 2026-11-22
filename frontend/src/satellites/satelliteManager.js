import * as Cesium from 'cesium';
import { twoline2satrec, propagate, gstime, eciToGeodetic, degreesLong, degreesLat } from 'satellite.js';
import { requestFrame } from '../core/renderGovernor.js';

/**
 * Satellite Manager for Blue Sentinel.
 * SGP4 orbital propagation for Indo-Pacific surveillance satellites
 * (GSAT-7 Naval Recon, ISS, Cartosat-3).
 * Enables real-time orbital paths and the "Satellite Direct View" (Nadir camera lock).
 */

const TLE_CATALOG = [
  {
    name: "GSAT-7 (RUKMINI) NAVAL RECON",
    norad: 39233,
    line1: "1 39233U 13044A   26085.12345678 -.00000123  00000-0  00000-0 0  9991",
    line2: "2 39233   4.1234  74.0000 0003500  45.0000 315.0000  1.00270000 45678",
    color: Cesium.Color.CYAN,
    altKm: 35786
  },
  {
    name: "ISS (ZARYA)",
    norad: 25544,
    line1: "1 25544U 98067A   26085.50000000  .00016717  00000-0  10270-3 0  9001",
    line2: "2 25544  51.6400  65.1234 0004500 120.0000 240.0000 15.50000000550001",
    color: Cesium.Color.GOLD,
    altKm: 420
  },
  {
    name: "CARTOSAT-3 (OPTICAL IMAGING)",
    norad: 44804,
    line1: "1 44804U 19081A   26085.20000000  .00001200  00000-0  45000-4 0  9992",
    line2: "2 44804  97.4500  78.2000 0012000  90.0000 270.0000 15.10000000340002",
    color: Cesium.Color.DEEPSKYBLUE,
    altKm: 505
  }
];

export class SatelliteManager {
  constructor(viewer) {
    this.viewer = viewer;
    this.satellites = [];
    this.pointPrimitives = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
    this.pathPrimitives = viewer.scene.primitives.add(new Cesium.PolylineCollection());
    this.selectedSatellite = null;
    this.active = true;
    this._initSats();
  }

  _initSats() {
    for (const satData of TLE_CATALOG) {
      const satrec = twoline2satrec(satData.line1, satData.line2);
      const point = this.pointPrimitives.add({
        position: Cesium.Cartesian3.ZERO,
        pixelSize: 8,
        color: satData.color,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2
      });

      this.satellites.push({
        ...satData,
        satrec,
        primitive: point,
        currentCartesian: null,
        currentLat: 0,
        currentLon: 0,
        currentAlt: 0
      });
    }

    this.updatePositions();
  }

  updatePositions() {
    const now = new Date();
    const gmst = gstime(now);

    for (const sat of this.satellites) {
      const posVel = propagate(sat.satrec, now);
      if (posVel.position) {
        const geodetic = eciToGeodetic(posVel.position, gmst);
        const lon = degreesLong(geodetic.longitude);
        const lat = degreesLat(geodetic.latitude);
        const alt = geodetic.height * 1000.0; // km to meters

        const cartesian = Cesium.Cartesian3.fromDegrees(lon, lat, alt);
        sat.primitive.position = cartesian;
        sat.currentCartesian = cartesian;
        sat.currentLat = lat;
        sat.currentLon = lon;
        sat.currentAlt = alt;
      }
    }
    requestFrame();
  }

  switchToNadirView(satName = "CARTOSAT-3 (OPTICAL IMAGING)") {
    const sat = this.satellites.find(s => s.name.includes(satName)) || this.satellites[0];
    if (!sat || !sat.currentCartesian) return;

    // Position camera at satellite coordinates, looking straight down (Nadir: pitch = -90 deg)
    this.viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(sat.currentLon, sat.currentLat, sat.currentAlt * 0.4),
      orientation: {
        heading: Cesium.Math.toRadians(0.0),
        pitch: Cesium.Math.toRadians(-89.9), // Nadir perspective
        roll: 0.0
      },
      duration: 2.0
    });

    console.log(`[Satellite Recon] Locked Nadir Direct View to ${sat.name}`);
  }

  setVisible(visible) {
    this.active = visible;
    this.pointPrimitives.show = visible;
    this.pathPrimitives.show = visible;
    requestFrame();
  }
}
