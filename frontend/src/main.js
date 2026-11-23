import * as Cesium from 'cesium';
import { installRenderGovernor, holdContinuousRender, requestFrame } from './core/renderGovernor.js';
import { applyShaderMode } from './shaders/shaderPipeline.js';
import { SatelliteManager } from './satellites/satelliteManager.js';
import { VesselManager } from './vessels/vesselManager.js';
import { SubmarineCablesLayer } from './layers/cablesLayer.js';
import { MarineBoundariesLayer } from './layers/boundariesLayer.js';
import { FirmsThermalLayer } from './layers/firmsLayer.js';
import { AuthManager } from './auth/authManager.js';

// Initialize Security Clearance & Authentication Manager
const authManager = new AuthManager();

// Setup Cesium Viewer with Keyless Esri World Imagery & Dark Sea Base
const viewer = new Cesium.Viewer('cesiumContainer', {
  baseLayer: false,
  animation: false,
  baseLayerPicker: false,
  fullscreenButton: false,
  geocoder: false,
  homeButton: false,
  infoBox: false,
  sceneModePicker: false,
  selectionIndicator: false,
  timeline: false,
  navigationHelpButton: false,
  scene3DOnly: true,
  shouldAnimate: true
});

// Asynchronously initialize Keyless Esri World Imagery with OSM fallback
async function initBaseImagery() {
  try {
    const provider = await Cesium.ArcGisMapServerImageryProvider.fromUrl(
      'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer',
      { enablePickFeatures: false }
    );
    viewer.imageryLayers.addImageryProvider(provider);
  } catch (err) {
    console.warn('[Imagery] Esri unavailable, falling back to OpenStreetMap:', err);
    const osmProvider = new Cesium.OpenStreetMapImageryProvider({
      url: 'https://tile.openstreetmap.org/'
    });
    viewer.imageryLayers.addImageryProvider(osmProvider);
  }
  requestFrame();
}
initBaseImagery();

// Atmospheric & Lighting tuning
viewer.scene.globe.enableLighting = true;
viewer.scene.globe.atmosphereBrightnessShift = -0.2;
viewer.scene.skyAtmosphere.show = true;

// Install Render Governor: Hard-caps rendering to 30 FPS max
installRenderGovernor(viewer, 30.0);
holdContinuousRender('core-sim');

// Initial Camera perspective: Centered over Indian Ocean / Indo-Pacific
viewer.camera.flyTo({
  destination: Cesium.Cartesian3.fromDegrees(78.0, 14.0, 3800000.0),
  orientation: {
    heading: Cesium.Math.toRadians(0.0),
    pitch: Cesium.Math.toRadians(-80.0),
    roll: 0.0
  },
  duration: 1.5
});

// Initialize System Modules
const cablesLayer = new SubmarineCablesLayer(viewer);
const boundariesLayer = new MarineBoundariesLayer(viewer);
const firmsLayer = new FirmsThermalLayer(viewer);
const satManager = new SatelliteManager(viewer);

// Target Inspector Update Callback
function handleSelectVessel(vessel) {
  const drawer = document.getElementById("inspector-drawer");
  const body = document.getElementById("drawer-body");
  if (!drawer || !body) return;

  const isHigh = vessel.risk_level === "HIGH";
  const isMed = vessel.risk_level === "MEDIUM";
  const barColor = isHigh ? '#ff2d55' : (isMed ? '#ffb340' : '#30d158');

  body.innerHTML = `
    <div>
      <div class="target-name">${vessel.name}</div>
      <div class="target-meta">MMSI: ${vessel.mmsi} · FLAG: ${vessel.flag} · ${vessel.vtype}</div>
    </div>

    <div class="risk-meter">
      <div class="risk-head">
        <span style="color:rgba(255,255,255,0.6)">LSTM ANOMALY RISK</span>
        <span style="display:flex; align-items:center; gap:6px;">
          ${isHigh ? '<img src="/assets/badge_icon.png" style="width:16px; height:16px; object-fit:contain;">' : ''}
          <b style="color:${barColor}">${vessel.risk_score}% [${vessel.risk_level}]</b>
        </span>
      </div>
      <div class="risk-bar">
        <div class="risk-bar-fill" style="width:${vessel.risk_score}%; background:${barColor}; box-shadow: 0 0 8px ${barColor}"></div>
      </div>
    </div>

    <div class="grid-metrics">
      <div class="metric-box">
        <div class="m-label">SPEED (SOG)</div>
        <div class="m-val">${vessel.sog} kn</div>
      </div>
      <div class="metric-box">
        <div class="m-label">HEADING (COG)</div>
        <div class="m-val">${vessel.cog}°</div>
      </div>
      <div class="metric-box">
        <div class="m-label">POSITION</div>
        <div class="m-val">${vessel.lat.toFixed(3)}N, ${vessel.lon.toFixed(3)}E</div>
      </div>
      <div class="metric-box">
        <div class="m-label">NEURAL PREDICTION</div>
        <div class="m-val" style="color:var(--cyan)">${vessel.predicted_lat?.toFixed(3) || '--'}N</div>
      </div>
    </div>

    <div class="flags-box">
      <div class="m-label" style="margin-bottom:4px;">EXPLAINABLE AI EVIDENCE</div>
      ${(vessel.flags || []).map(f => `<span class="flag-chip">⚠ ${f}</span>`).join('')}
    </div>

    <button class="rail-btn action-btn" id="btn-lock-cam" style="margin-top:14px; text-align:center; width:100%;">
      🎯 RE-CENTER CAMERA LOCK
    </button>
  `;

  document.getElementById("btn-lock-cam")?.addEventListener('click', () => {
    vesselManager.selectVessel(vessel.mmsi);
  });
}

const vesselManager = new VesselManager(viewer, handleSelectVessel);

// Periodic Satellite Orbit propagation
setInterval(() => {
  satManager.updatePositions();
}, 1000);

// UI Event Handlers: Shader Switches
document.querySelectorAll('[data-shader]').forEach(btn => {
  btn.addEventListener('click', (e) => {
    document.querySelectorAll('[data-shader]').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    const mode = e.target.getAttribute('data-shader');
    applyShaderMode(viewer, mode);
  });
});

// UI Event Handlers: Layer Toggles
document.getElementById('layer-vessels')?.addEventListener('change', (e) => {
  vesselManager.setVesselsVisible(e.target.checked);
});
document.getElementById('layer-trails')?.addEventListener('change', (e) => {
  vesselManager.setTrailsVisible(e.target.checked);
});
document.getElementById('layer-satellites')?.addEventListener('change', (e) => {
  satManager.setVisible(e.target.checked);
});
document.getElementById('layer-cables')?.addEventListener('change', (e) => {
  cablesLayer.setVisible(e.target.checked);
});
document.getElementById('layer-boundaries')?.addEventListener('change', (e) => {
  boundariesLayer.setVisible(e.target.checked);
});
document.getElementById('layer-firms')?.addEventListener('change', (e) => {
  firmsLayer.setVisible(e.target.checked);
});

// UI Event Handlers: Camera Modes
document.getElementById('btn-nadir-sat')?.addEventListener('click', () => {
  satManager.switchToNadirView();
});

document.getElementById('btn-reset-cam')?.addEventListener('click', () => {
  vesselManager.clearSelection();
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(78.0, 14.0, 3800000.0),
    orientation: {
      heading: 0.0,
      pitch: Cesium.Math.toRadians(-80.0),
      roll: 0.0
    },
    duration: 1.5
  });
});

document.getElementById('btn-close-drawer')?.addEventListener('click', () => {
  vesselManager.clearSelection();
  const body = document.getElementById("drawer-body");
  if (body) body.innerHTML = '<div class="empty-state">SELECT AN ASSET TO INITIATE TARGET LOCK</div>';
});

// Live UTC Clock
setInterval(() => {
  const clock = document.getElementById('utc-clock');
  if (clock) {
    clock.innerText = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  }
}, 1000);

console.log("[Blue Sentinel] 3D C4ISR Web Console Online.");
