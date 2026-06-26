// shaders/plasma.frag
// Renders a screen-space hot plasma scattering glow.
// Composited additively over the scene after the lighting pass.
//
// Models:
//   Thomson scattering: σ_T = 6.65×10⁻²⁹ m² (e-γ scattering)
//   Mean free path: λ = 1/(n_e × σ_T) — universe opaque below recombination
//   Optical depth:  τ = ∫ n_e σ_T dl — controls opacity of plasma glow
//
// Visual:
//   - Dense hot regions glow bright orange-white (T > 100 MeV)
//   - Cooling regions shift to red then dim (T → Λ_QCD)
//   - Plasma ripples: pressure waves from particle interactions
//   - Electroweak sector: faint purple haze at T > 100 GeV

#version 410 core

in vec2 v_uv;

uniform sampler2D u_scene;        // lit scene
uniform sampler2D u_plasma_noise; // pre-baked 3D noise slice
uniform float u_time;
uniform float u_T_GeV;            // cosmic temperature
uniform int   u_epoch;
uniform float u_opacity;          // overall plasma opacity (0 at recombination)

out vec4 frag_color;

// ── Blackbody plasma colour ────────────────────────────────────
vec3 plasma_colour(float T_GeV) {
    // Map temperature to visual colour:
    // T > 100 GeV: white-blue (QGP / electroweak)
    // T ~ 0.2 GeV: orange-red (QCD transition)
    // T < 0.01 GeV: deep red, fading
    if      (T_GeV > 10.0)  return mix(vec3(0.85,0.90,1.00), vec3(1.0,1.0,1.0), min(1.0, T_GeV/100.0));
    else if (T_GeV > 0.5)   return mix(vec3(1.0, 0.55, 0.10), vec3(0.85,0.90,1.00), (T_GeV-0.5)/9.5);
    else if (T_GeV > 0.15)  return mix(vec3(0.95, 0.20, 0.05), vec3(1.0, 0.55, 0.10), (T_GeV-0.15)/0.35);
    else                     return vec3(0.5, 0.05, 0.01) * max(0.0, T_GeV/0.15);
}

// ── Screen-space plasma density ───────────────────────────────
float plasma_density(vec2 uv) {
    // Layered noise for turbulent plasma
    float t = u_time;
    float n1 = texture(u_plasma_noise, uv * 2.1 + vec2(t*0.07, t*0.03)).r;
    float n2 = texture(u_plasma_noise, uv * 4.3 - vec2(t*0.05, t*0.09)).r;
    float n3 = texture(u_plasma_noise, uv * 8.7 + vec2(t*0.12, -t*0.04)).r;
    return n1*0.5 + n2*0.3 + n3*0.2;
}

// ── Pressure ripple waves ─────────────────────────────────────
// Each major particle interaction sends a pressure wave outward.
// We approximate this with animated concentric rings.
float pressure_waves(vec2 uv) {
    vec2  c1  = vec2(0.3 + 0.2*sin(u_time*0.7),  0.5 + 0.15*cos(u_time*0.5));
    vec2  c2  = vec2(0.7 - 0.2*cos(u_time*0.4),  0.3 + 0.20*sin(u_time*0.8));
    float r1  = length(uv - c1);
    float r2  = length(uv - c2);
    float w1  = sin((r1 - u_time*0.15) * 30.0) * exp(-r1 * 3.5);
    float w2  = sin((r2 - u_time*0.12) * 25.0) * exp(-r2 * 3.0);
    return clamp((w1 + w2) * 0.5, 0.0, 1.0);
}

// ── Thomson scattering haze ───────────────────────────────────
// Bright scene pixels scatter light into a Gaussian halo.
// Approximated by a multi-tap blur weighted by plasma density.
vec3 scatter_haze(sampler2D scene, vec2 uv, float density) {
    const int TAPS = 8;
    const float RADIUS = 0.018;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < TAPS; i++) {
        float a  = float(i) / float(TAPS) * 6.28318;
        vec2  o  = vec2(cos(a), sin(a)) * RADIUS * density;
        acc     += texture(scene, uv + o).rgb;
    }
    return acc / float(TAPS);
}

void main() {
    // No plasma glow after recombination (epoch 6+)
    if (u_epoch >= 6 || u_opacity < 0.01) {
        frag_color = texture(u_scene, v_uv);
        return;
    }

    vec3  scene   = texture(u_scene, v_uv).rgb;
    float density = plasma_density(v_uv);
    float waves   = pressure_waves(v_uv);
    vec3  pcol    = plasma_colour(u_T_GeV);

    // Thomson scatter haze: nearby bright pixels bleed
    vec3  haze    = scatter_haze(u_scene, v_uv, density * 0.8);

    // Plasma emission: dense hot regions emit blackbody radiation
    float emission = density * u_opacity;
    vec3  plasma   = pcol * emission * 0.35;

    // Pressure wave tint
    plasma += pcol * waves * 0.15 * u_opacity;

    // Electroweak haze: at T > 100 GeV, purple EW symmetry haze
    if (u_T_GeV > 10.0 && u_epoch <= 2) {
        float ew_str = min(1.0, u_T_GeV / 200.0);
        plasma += vec3(0.30, 0.10, 0.55) * density * ew_str * 0.25;
    }

    // Compose: scene + scattered haze + plasma emission
    vec3  result  = scene
                  + haze   * density * u_opacity * 0.20
                  + plasma;

    frag_color = vec4(result, 1.0);
}
