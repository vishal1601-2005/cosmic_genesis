// shaders/post/godrays.frag
// Screen-Space God Rays (Radial Light Shafts).
//
// Based on the technique by Kenny Mitchell (Crytek):
// March from pixel toward the light source in screen space,
// accumulating occlusion samples weighted by distance falloff.
//
// Each epoch has its own "light source" position and colour:
//   Epoch 0 (strings):      No directional light — ambient foam glow
//   Epoch 1 (inflation):    Central bright point (inflaton false vacuum)
//   Epoch 2 (baryogenesis): Multiple hot plasma sources
//   Epoch 3 (QCD):          Confinement glow from hadronic clusters
//   Epoch 5 (BBN):          Photon bath — omnidirectional
//   Epoch 6 (recombination):CMB last-scattering surface as light dome
//   Epoch 7 (structure):    Galaxy centres / AGN as point sources

#version 410 core

in vec2 v_uv;

uniform sampler2D u_scene;       // scene colour (after bloom)
uniform sampler2D u_occlusion;   // occlusion mask (bright pixels = emissive)
uniform vec2  u_light_pos_ss;    // light source in screen space [0,1]
uniform vec3  u_light_color;     // shaft colour
uniform float u_density;         // sample density (default 0.96)
uniform float u_weight;          // per-sample weight (default 0.015)
uniform float u_decay;           // exponential decay per sample (default 0.97)
uniform float u_exposure;        // final exposure multiplier (default 0.6)
uniform int   u_n_samples;       // number of radial samples (default 80)
uniform float u_time;
uniform int   u_epoch;

out vec4 frag_color;

void main() {
    // Direction from pixel toward light source
    vec2 delta = v_uv - u_light_pos_ss;
    delta *= u_density / float(u_n_samples);

    vec2  tc     = v_uv;
    float decay  = 1.0;
    vec3  result = vec3(0.0);

    // Radial march
    for (int i = 0; i < u_n_samples; i++) {
        tc       -= delta;
        vec3 samp = texture(u_occlusion, tc).rgb;

        // Weight by how bright the sample is
        float lum = dot(samp, vec3(0.299, 0.587, 0.114));
        result   += samp * decay * u_weight * lum;
        decay    *= u_decay;
    }

    // Epoch-specific colour tint on god rays
    vec3 shaft_col = u_light_color;
    if (u_epoch == 0) shaft_col = vec3(0.5, 0.3, 1.0);
    else if (u_epoch == 1) shaft_col = vec3(0.9, 0.6, 1.0);
    else if (u_epoch == 2) shaft_col = vec3(1.0, 0.5, 0.1);
    else if (u_epoch == 6) shaft_col = vec3(1.0, 0.85, 0.6);
    else if (u_epoch == 7) shaft_col = vec3(0.4, 0.6, 1.0);

    result *= shaft_col * u_exposure;

    // Additive blend with scene
    vec3 scene = texture(u_scene, v_uv).rgb;
    frag_color = vec4(scene + result, 1.0);
}
