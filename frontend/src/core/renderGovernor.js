/**
 * Render Governor for Blue Sentinel CesiumJS Viewport.
 * Hard-caps rendering to maximum 30 FPS to optimize GPU memory and battery consumption.
 * Activates requestRenderMode when idle, rendering only when camera or telemetry mutates.
 */

let _viewer = null;
let _targetFPS = 30.0;
let _frameIntervalMs = 1000.0 / 30.0;
let _lastRenderTime = 0;
const _holds = new Set();

export function installRenderGovernor(viewer, targetFPS = 30.0) {
  _viewer = viewer;
  _targetFPS = targetFPS;
  _frameIntervalMs = 1000.0 / targetFPS;

  const scene = viewer.scene;
  
  // Hard cap frame rate on Cesium viewer
  viewer.targetFrameRate = targetFPS;
  
  // Set request render mode
  scene.requestRenderMode = true;
  scene.maximumRenderTimeChange = Infinity;

  // Custom throttled loop
  viewer.clock.onTick.addEventListener(() => {
    const now = performance.now();
    if (now - _lastRenderTime >= _frameIntervalMs) {
      _lastRenderTime = now;
      if (_holds.size > 0) {
        scene.requestRender();
      }
    }
  });

  console.log(`[Render Governor] Active: Hard-capped at ${_targetFPS} FPS.`);
}

export function holdContinuousRender(ownerId) {
  if (!ownerId) return;
  _holds.add(ownerId);
  if (_viewer?.scene) {
    _viewer.scene.requestRender();
  }
}

export function releaseContinuousRender(ownerId) {
  if (!ownerId) return;
  _holds.delete(ownerId);
}

export function requestFrame() {
  if (_viewer?.scene) {
    _viewer.scene.requestRender();
  }
}
