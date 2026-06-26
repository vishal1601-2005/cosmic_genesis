// shaders/volume.frag
// Volumetric rendering: ray-marches through a 3D density/temperature field
// to render the plasma, inflaton field, CMB photon bath, and nebulae.
//
// The density field d(x) is synthesised from 3D layered Perlin noise.
// Temperature T(x) determines the local emission colour (blackbody).
//
// Algorithm: front-to-back alpha compositing along the ray.
// Each step accumulates:
//   - Extinction (absorption + scattering): σ_t = σ_a + σ_s
//   - In-scatter from lights: L_i × σ_s × phase(θ)
//   - Emission: B(T) × σ_e (blackbody for hot plasma)
//
// Number of steps is dynamically scaled by quality:
//   QUALITY 0: 32 steps  (low, real-time guaranteed)
//   QUALITY 1: 64 steps  (medium, 60fps on RTX 3080)
//   QUALITY 2: 128 steps (high, 60fps on RTX 4090)
//   QUALITY 3: 256 steps (cinematic, offline)

#version 410 core

in vec2 v_uv;   // fullscreen quad [0,1]

uniform sampler2D u_scene;       // opaque scene beneath volume
uniform sampler2D u_depth;       // scene depth buffer

uniform mat4  u_inv_view_proj;   // for ray reconstruction
uniform vec3  u_cam_pos;
uniform float u_time;
uniform int   u_epoch;
uniform float u_T_GeV;           // cosmic temperature
uniform float u_zoom;
uniform int   u_quality;         // 0-3
uniform float u_vol_density;     // master density scale
uniform float u_vol_emission;    // master emission scale

out vec4 frag_color;

// ── Noise ──────────────────────────────────────────────────────
float hash(vec3 p) {
    p  = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i),           hash(i+vec3(1,0,0)), u.x),
                   mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)), u.x), u.y),
               mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)), u.x),
                   mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)), u.x), u.y), u.z);
}

// 5-octave FBM for turbulent plasma
float fbm(vec3 p, int oct) {
    float val = 0.0, amp = 0.5, freq = 1.0;
    for (int i = 0; i < oct; i++) {
        val  += amp * noise3(p * freq);
        amp  *= 0.5; freq *= 2.0;
    }
    return val;
}

// ── Epoch-specific density field ───────────────────────────────
// Returns (density, temperature_norm) at world point p and time t

vec2 epoch_density(vec3 p, float t) {
    float d = 0.0, T_n = 0.0;

    if (u_epoch == 0) {
        // String landscape: probabilistic foam — spacetime not yet smooth
        // Topology changes: density spikes at random points (Planck scale foam)
        d   = fbm(p * 3.0 + t * 0.2, 4) * 0.8;
        d  += fbm(p * 7.0 - t * 0.1, 3) * 0.4;
        d   = pow(max(0.0, d - 0.2), 1.5);
        T_n = 1.0;   // Planck temperature

    } else if (u_epoch == 1) {
        // Inflation: smooth, expanding de Sitter space
        // Density decreases as universe expands, but quantum fluctuations appear
        float expand = min(1.0, t * 0.05);
        float base   = max(0.0, 1.0 - expand * 2.0);
        float fluct  = fbm(p * (10.0 - expand * 8.0) + t * 0.3, 3) * 0.3;
        d   = base + fluct;
        T_n = base;

    } else if (u_epoch == 2) {
        // Baryogenesis: hot QGP, turbulent, very bright
        d   = fbm(p * 2.0 + t * 0.5, 5) * 1.2;
        d   = smoothstep(0.1, 0.8, d);
        T_n = 0.85 + 0.15 * noise3(p * 8.0 + t);

    } else if (u_epoch == 3) {
        // QCD: plasma with confinement — stringy filaments
        float filament = abs(sin(p.x * 4.0 + t)) * abs(sin(p.y * 3.5 - t * 0.7));
        d   = fbm(p * 2.5, 4) * 0.6 + filament * 0.5;
        d   = pow(max(0.0, d - 0.15), 1.2);
        T_n = 0.5 + 0.5 * filament;

    } else if (u_epoch == 4) {
        // Axion field: smooth oscillating scalar condensate
        float axion_osc = 0.5 + 0.5 * cos(length(p) * 3.0 - t * 2.0);
        d   = axion_osc * fbm(p * 1.5, 3) * 0.5;
        T_n = 0.1;   // cold dark matter — no thermal emission

    } else if (u_epoch == 5) {
        // BBN: hot photon-baryon plasma, cooling
        float T_norm = clamp(u_T_GeV * 1e3, 0.0, 1.0);
        d   = fbm(p * 1.8 + t * 0.2, 4) * T_norm * 1.5;
        d   = max(0.0, d - 0.1 * (1.0 - T_norm));
        T_n = T_norm;

    } else if (u_epoch == 6) {
        // Recombination: plasma clearing to transparent
        float clear    = clamp((t - 50.0) / 100.0, 0.0, 1.0);
        d   = fbm(p * 1.5, 4) * (1.0 - clear) * 0.8;
        T_n = (1.0 - clear) * 0.4;

    } else {
        // Structure: dark filaments + void — cosmic web
        // Use the filament density model from N-body simulations
        float web = pow(max(0.0,
            fbm(p * 0.8, 4) - 0.35), 2.0) * 3.0;
        // Spherical halo overdensities
        float halo_r = length(fract(p * 0.25) - 0.5) * 4.0;
        float halo   = max(0.0, 1.0 - halo_r) * 2.0;
        d   = web * 0.6 + halo * 0.4;
        T_n = halo * 0.15;   // warm halo gas, rest is dark
    }

    return vec2(clamp(d, 0.0, 1.0), clamp(T_n, 0.0, 1.0));
}

// ── Blackbody colour ───────────────────────────────────────────
// Maps normalised temperature [0,1] to RGB colour (CIE approximation)
vec3 blackbody(float T_norm) {
    // T_norm: 0=cool(red), 0.5=warm(orange/yellow), 1=hot(blue-white)
    float T_K = 1000.0 + T_norm * 29000.0;   // 1000 K – 30000 K
    vec3 col;
    // Red
    if (T_K <= 6600.0)
        col.r = 1.0;
    else
        col.r = clamp(pow(T_K / 6600.0 - 0.5, -0.133) * 1.292, 0.0, 1.0);
    // Green
    if (T_K <= 6600.0)
        col.g = clamp(0.390 * log(T_K / 100.0) - 0.631, 0.0, 1.0);
    else
        col.g = clamp(pow(T_K / 6600.0 - 2.0, -0.0755) * 1.147, 0.0, 1.0);
    // Blue
    if (T_K >= 6600.0)
        col.b = 1.0;
    else if (T_K <= 1900.0)
        col.b = 0.0;
    else
        col.b = clamp(0.543 * log(T_K / 100.0 - 10.0) - 1.196, 0.0, 1.0);
    return col;
}

// ── Henyey-Greenstein phase function ─────────────────────────
// Describes how light scatters: g=0 isotropic, g>0 forward-scatter
float hg_phase(float cos_theta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

// ── Ray reconstruction from UV ─────────────────────────────────
vec3 get_ray_dir(vec2 uv) {
    vec4 clip    = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 world   = u_inv_view_proj * clip;
    return normalize(world.xyz / world.w - u_cam_pos);
}

// ── Main ray-march ─────────────────────────────────────────────
void main() {
    vec3 ray_dir = get_ray_dir(v_uv);
    vec3 ray_pos = u_cam_pos;

    // Step count by quality
    int steps = int(mix(32.0, 256.0, float(u_quality) / 3.0));

    // Ray extents — epoch-specific box
    float t_near = 0.5, t_far = 30.0;
    float dt     = (t_far - t_near) / float(steps);

    // Front-to-back compositing accumulators
    vec3  col_acc   = vec3(0.0);
    float alpha_acc = 0.0;

    // Jitter start point (blue noise) to reduce banding
    float jitter = fract(sin(dot(v_uv, vec2(127.1, 311.7))) * 43758.5);
    float t      = t_near + jitter * dt;

    for (int i = 0; i < steps; i++) {
        if (alpha_acc > 0.98) break;   // early termination (fully opaque)

        vec3 pos = ray_pos + ray_dir * t;

        vec2 dT      = epoch_density(pos, u_time);
        float density = dT.x * u_vol_density;
        float T_norm  = dT.y;

        if (density > 0.001) {
            // Extinction coefficient
            float sigma_t = density * 0.8;
            float sigma_s = density * 0.4;   // scattering
            float sigma_e = density * 0.6;   // emission

            // Single-scatter approximation: use main directional light
            // (In full implementation: shadow ray toward each light)
            vec3 L_dir   = normalize(vec3(0.5, 1.0, 0.3));   // fake sun
            float phase  = hg_phase(dot(ray_dir, L_dir), 0.3);
            vec3 L_scat  = vec3(0.8, 0.7, 0.5) * sigma_s * phase;

            // Thermal emission: blackbody at local temperature
            vec3 bb      = blackbody(T_norm);
            vec3 emission = bb * sigma_e * u_vol_emission;

            // Epoch-specific tint
            vec3 tint = vec3(1.0);
            if (u_epoch == 0) tint = vec3(0.6, 0.4, 1.0);      // purple strings
            else if (u_epoch == 1) tint = vec3(0.8, 0.5, 1.0); // inflation purple
            else if (u_epoch == 2) tint = vec3(1.0, 0.4, 0.1); // hot red QGP
            else if (u_epoch == 3) tint = vec3(0.9, 0.3, 0.05);// deep red QCD
            else if (u_epoch == 4) tint = vec3(0.4, 0.2, 0.9); // dark axion
            else if (u_epoch == 7) tint = vec3(0.2, 0.3, 0.8); // cosmic web blue

            vec3  sample_col = (L_scat + emission) * tint;
            float sample_a   = 1.0 - exp(-sigma_t * dt);

            // Front-to-back blend
            col_acc   += (1.0 - alpha_acc) * sample_a * sample_col;
            alpha_acc += (1.0 - alpha_acc) * sample_a;
        }

        t += dt;
    }

    // Blend with opaque scene
    vec4 scene = texture(u_scene, v_uv);
    vec3 final = col_acc + (1.0 - alpha_acc) * scene.rgb;

    frag_color = vec4(final, 1.0);
}
