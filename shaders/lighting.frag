// shaders/lighting.frag
// Deferred PBR lighting pass.
// Reads the 4 G-Buffer textures and computes full PBR shading.
//
// Lighting model:
//   - Cook-Torrance GGX specular BRDF
//   - Lambertian diffuse
//   - Multiple point lights (one per nearby particle — self-emission as light)
//   - Ambient IBL (Image-Based Lighting) from a precomputed env map
//     approximated here as a spherical harmonic sky model
//   - Emission added directly (HDR, no clamping)
//
// Output is HDR float16 colour ready for bloom + tonemap.

#version 410 core

in vec2 v_uv;   // fullscreen quad [0,1]

// G-Buffer samplers
uniform sampler2D gAlbedoMetallic;
uniform sampler2D gNormalRoughness;
uniform sampler2D gEmission;
uniform sampler2D gPositionDepth;

// Environment
uniform vec3  u_cam_pos;
uniform float u_time;
uniform int   u_epoch;         // 0-7
uniform float u_T_GeV;         // temperature
uniform float u_ambient_str;   // ambient light strength (varies by epoch)

// Up to 32 dynamic lights (brightest nearby particles)
uniform int   u_n_lights;
uniform vec3  u_light_pos[32];
uniform vec3  u_light_col[32];
uniform float u_light_str[32];  // HDR strength

out vec4 frag_color;   // HDR output

// ── PBR functions ──────────────────────────────────────────────
const float PI = 3.14159265359;

// GGX Normal Distribution Function
float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

// Geometry: Smith-Schlick GGX
float G_SmithSchlick(float NdotV, float NdotL, float roughness) {
    float r  = roughness + 1.0;
    float k  = (r * r) / 8.0;
    float gv = NdotV / (NdotV * (1.0 - k) + k + 1e-7);
    float gl = NdotL / (NdotL * (1.0 - k) + k + 1e-7);
    return gv * gl;
}

// Fresnel: Schlick approximation
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Full Cook-Torrance BRDF
vec3 BRDF(vec3 N, vec3 V, vec3 L,
          vec3 albedo, float metallic, float roughness,
          vec3 light_color, float light_str) {
    vec3  H      = normalize(V + L);
    float NdotL  = max(dot(N, L), 0.0);
    float NdotV  = max(dot(N, V), 0.001);
    float NdotH  = max(dot(N, H), 0.0);
    float HdotV  = max(dot(H, V), 0.0);

    // F0: base reflectance
    vec3  F0     = mix(vec3(0.04), albedo, metallic);

    float D      = D_GGX(NdotH, roughness);
    float G      = G_SmithSchlick(NdotV, NdotL, roughness);
    vec3  F      = F_Schlick(HdotV, F0);

    // Specular
    vec3  spec   = (D * G * F) / (4.0 * NdotV * NdotL + 1e-7);

    // Diffuse: energy conserving
    vec3  kd     = (1.0 - F) * (1.0 - metallic);
    vec3  diff   = kd * albedo / PI;

    return (diff + spec) * light_color * light_str * NdotL;
}

// ── Epoch sky / ambient model ──────────────────────────────────
// Returns the ambient IBL approximation for the current epoch.
// Each epoch has a distinct "sky" colour and structure.
vec3 epoch_ambient(vec3 N) {
    vec3 sky;
    if (u_epoch == 0) {
        // String epoch: deep violet, slight 10D shimmer
        float sh = 0.5 + 0.5 * sin(dot(N, vec3(1.3, 2.7, 1.9)) * 4.0 + u_time * 0.5);
        sky = mix(vec3(0.04, 0.02, 0.12), vec3(0.12, 0.08, 0.28), sh);
    } else if (u_epoch == 1) {
        // Inflation: uniform bright background expanding
        float expand = min(1.0, (u_time * 0.1));
        sky = mix(vec3(0.02, 0.01, 0.06), vec3(0.3, 0.15, 0.45), expand);
    } else if (u_epoch == 2) {
        // Baryogenesis: hot orange-red plasma
        sky = vec3(0.15, 0.04, 0.02) * (1.0 + 0.3 * sin(u_time * 2.0));
    } else if (u_epoch == 3) {
        // QCD: deep red cooling plasma
        sky = vec3(0.08, 0.02, 0.01) * (1.5 - u_T_GeV * 0.5);
    } else if (u_epoch == 4) {
        // Axion: dim purple (dark sector)
        sky = vec3(0.02, 0.01, 0.06);
    } else if (u_epoch == 5) {
        // BBN: warm blue-white (hot gas, photon bath)
        float T_norm = clamp(u_T_GeV * 1e3, 0.0, 1.0);
        sky = mix(vec3(0.02, 0.04, 0.12), vec3(0.15, 0.20, 0.35), T_norm);
    } else if (u_epoch == 6) {
        // Recombination: warm amber fading to dark
        sky = vec3(0.12, 0.06, 0.02) * max(0.0, 1.0 - u_time * 0.05);
    } else {
        // Structure: near-black with faint starfield shimmer
        float star = step(0.998, fract(sin(dot(N.xy, vec2(127.1, 311.7))) * 43758.5));
        sky = vec3(0.005, 0.005, 0.012) + star * vec3(0.8, 0.85, 1.0);
    }
    // Hemisphere: brighter top
    float hemi = 0.5 + 0.5 * N.y;
    return sky * mix(0.5, 1.0, hemi) * u_ambient_str;
}

// ── Main ───────────────────────────────────────────────────────
void main() {
    // Sample G-Buffer
    vec4  albedo_met  = texture(gAlbedoMetallic,  v_uv);
    vec4  normal_rou  = texture(gNormalRoughness,  v_uv);
    vec4  emission_a  = texture(gEmission,         v_uv);
    vec4  pos_depth   = texture(gPositionDepth,    v_uv);

    vec3  albedo      = albedo_met.rgb;
    float metallic    = albedo_met.a;
    vec3  N           = normalize(normal_rou.rgb * 2.0 - 1.0);
    float roughness   = clamp(normal_rou.a, 0.04, 1.0);
    vec3  emission    = emission_a.rgb;
    float alpha       = emission_a.a;
    vec3  world_pos   = pos_depth.rgb;

    vec3  V           = normalize(u_cam_pos - world_pos);

    // If no geometry written (background): output epoch sky
    if (pos_depth.a >= 1.0) {
        vec3 ray = normalize(world_pos - u_cam_pos);
        frag_color = vec4(epoch_ambient(ray) * 2.5, 1.0);
        return;
    }

    // ── Ambient IBL (simplified SH) ──────────────────────
    vec3 Lo = epoch_ambient(N);

    // ── Dynamic lights from nearby emissive particles ──────
    for (int i = 0; i < u_n_lights && i < 32; i++) {
        vec3  L_dir  = u_light_pos[i] - world_pos;
        float dist2  = dot(L_dir, L_dir) + 0.001;
        float atten  = u_light_str[i] / dist2;   // inverse square falloff
        if (atten < 0.001) continue;
        Lo += BRDF(N, V, normalize(L_dir),
                   albedo, metallic, roughness,
                   u_light_col[i], atten);
    }

    // ── Emission: add directly (HDR — no clamping) ────────
    vec3 color = Lo + emission;

    frag_color = vec4(color, alpha);
}
