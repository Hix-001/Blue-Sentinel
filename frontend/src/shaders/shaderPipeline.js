import * as Cesium from 'cesium';

/**
 * GLSL Tactical Sensor Shaders ported from God's Eye View:
 * 1. FLIR Thermal (Ironbow "Predator" + White-Hot)
 * 2. CRT Surveillance CCTV (analog scanlines & noise)
 * 3. Night Vision (NVG Phosphor Green)
 * 4. Noir Tactical
 */

const FLIR_FRAGMENT = `
  uniform sampler2D colorTexture;
  uniform float intensity;
  in vec2 v_textureCoordinates;

  vec3 ironbow(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.0, 0.0, 0.0);
    const vec3 c1 = vec3(0.13, 0.0, 0.30);
    const vec3 c2 = vec3(0.49, 0.0, 0.45);
    const vec3 c3 = vec3(0.86, 0.10, 0.18);
    const vec3 c4 = vec3(1.0, 0.55, 0.0);
    const vec3 c5 = vec3(1.0, 0.91, 0.32);
    const vec3 c6 = vec3(1.0, 1.0, 1.0);
    float s = t * 6.0;
    if (s < 1.0) return mix(c0, c1, s);
    if (s < 2.0) return mix(c1, c2, s - 1.0);
    if (s < 3.0) return mix(c2, c3, s - 2.0);
    if (s < 4.0) return mix(c3, c4, s - 3.0);
    if (s < 5.0) return mix(c4, c5, s - 4.0);
    return mix(c5, c6, s - 5.0);
  }

  void main() {
    vec4 color = texture(colorTexture, v_textureCoordinates);
    float luma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    
    // Invert & enhance thermal gradient
    float temp = pow(luma, 1.3);
    vec3 thermal = ironbow(temp);
    
    // Scanline & noise effect
    float scanline = sin(v_textureCoordinates.y * 700.0) * 0.04;
    thermal -= scanline;

    out_FragColor = vec4(mix(color.rgb, thermal, intensity), color.a);
  }
`;

const SURVEILLANCE_FRAGMENT = `
  uniform sampler2D colorTexture;
  uniform float intensity;
  in vec2 v_textureCoordinates;

  void main() {
    vec2 uv = v_textureCoordinates;
    vec4 color = texture(colorTexture, uv);

    // Green-tinted security monitor
    float luma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    vec3 cctv = vec3(luma * 0.35, luma * 1.05, luma * 0.45);

    // CRT scanlines
    float scan = sin(uv.y * 650.0) * 0.12;
    cctv -= scan;

    // Corner Vignette
    vec2 vig = uv * (1.0 - uv);
    cctv *= pow(vig.x * vig.y * 15.0, 0.25);

    out_FragColor = vec4(mix(color.rgb, cctv, intensity), color.a);
  }
`;

const NOIR_FRAGMENT = `
  uniform sampler2D colorTexture;
  uniform float intensity;
  in vec2 v_textureCoordinates;

  void main() {
    vec2 uv = v_textureCoordinates;
    vec4 color = texture(colorTexture, uv);

    // High-contrast monochromatic
    float luma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    float highContrast = clamp((luma - 0.5) * 1.6 + 0.5, 0.0, 1.0);
    vec3 gray = vec3(highContrast);

    out_FragColor = vec4(mix(color.rgb, gray, intensity), color.a);
  }
`;

let _activeStage = null;

export function applyShaderMode(viewer, mode = 'normal') {
  const collection = viewer.scene.postProcessStages;
  
  if (_activeStage) {
    collection.remove(_activeStage);
    _activeStage = null;
  }

  if (mode === 'normal') {
    viewer.scene.requestRender();
    return;
  }

  let glslCode = null;
  if (mode === 'thermal') glslCode = FLIR_FRAGMENT;
  else if (mode === 'surveillance') glslCode = SURVEILLANCE_FRAGMENT;
  else if (mode === 'noir') glslCode = NOIR_FRAGMENT;

  if (glslCode) {
    _activeStage = new Cesium.PostProcessStage({
      fragmentShader: glslCode,
      uniforms: {
        intensity: 1.0
      }
    });
    collection.add(_activeStage);
  }

  viewer.scene.requestRender();
  console.log(`[Shader Pipeline] Switched sensor mode to: ${mode}`);
}
