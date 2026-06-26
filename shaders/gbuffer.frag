// shaders/gbuffer.frag
// Writes to 4 MRT targets:
//   layout 0: gAlbedoMetallic    — RGB=albedo,      A=metallic
//   layout 1: gNormalRoughness   — RGB=world normal, A=roughness
//   layout 2: gEmission          — RGB=emission×strength (HDR)
//   layout 3: gPositionDepth     — RGB=world position, A=linear depth
//
// Particle-type-specific surface detail is baked here so the lighting
// pass is simple and fast.

#version 410 core

in vec3  v_world_pos;
in vec3  v_normal;
in vec2  v_uv;          // billboard UV in [-1,1]
in vec3  v_albedo;
in vec3  v_emission;
in float v_emission_str;
in float v_metallic;
in float v_roughness;
in float v_age;
flat in int v_type;

uniform float u_time;
uniform float u_zoom;    // 0=cosmic, 1=Planck scale
uniform int   u_epoch;
uniform float u_near;    // camera near plane
uniform float u_far;     // camera far plane

layout(location = 0) out vec4 gAlbedoMetallic;
layout(location = 1) out vec4 gNormalRoughness;
layout(location = 2) out vec4 gEmission;
layout(location = 3) out vec4 gPositionDepth;

// ── Utility ────────────────────────────────────────────────────
float sdCircle(vec2 p) { return length(p); }

// 2D hash for procedural detail
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

float noise2(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = dot(hash2(i),              f);
    float b = dot(hash2(i + vec2(1,0)),  f - vec2(1,0));
    float c = dot(hash2(i + vec2(0,1)),  f - vec2(0,1));
    float d = dot(hash2(i + vec2(1,1)),  f - vec2(1,1));
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

// ── Type-specific surface shaders ─────────────────────────────
//
// Type 0 = generic hadron/atom
// Type 1 = quark (colour charge shimmer)
// Type 2 = gluon (octet rotating pattern)
// Type 3 = photon (oscillating EM wave rings)
// Type 4 = dark matter (ghostly, nearly invisible)
// Type 5 = string cross-section (worldsheet)
// Type 6 = nucleus (layered shells, doubly magic He-4 = brightest)
// Type 7 = dark matter halo (massive, dim volume)
// Type 8 = star (blackbody emission, limb darkening)
// Type 9 = galaxy merger shockwave

struct Surface {
    vec3  albedo;
    vec3  emission;
    float metallic;
    float roughness;
    float alpha;
};

Surface surface_quark(vec2 uv) {
    float d = length(uv);
    if (d > 1.0) discard;

    // Colour charge: 3 rotating spokes (SU(3) triplet)
    float angle  = atan(uv.y, uv.x) + u_time * 2.2;
    float spoke  = pow(max(0.0, cos(angle * 3.0)), 6.0);
    float core   = smoothstep(0.3, 0.0, d);
    float corona = smoothstep(1.0, 0.2, d) * (1.0 - core);

    // Three RGB channels for R/G/B colour charges
    vec3 charge_col = vec3(
        pow(max(0.0, cos(angle * 3.0 + 0.0)),        6.0),
        pow(max(0.0, cos(angle * 3.0 + 2.094395)),   6.0),
        pow(max(0.0, cos(angle * 3.0 + 4.188790)),   6.0)
    );

    vec3 albedo   = mix(v_albedo, charge_col, 0.45) * corona;
    vec3 emission = charge_col * spoke * 2.5 + v_albedo * core * 4.0;

    return Surface(albedo, emission, 0.0, 0.4, smoothstep(1.0, 0.85, d));
}

Surface surface_gluon(vec2 uv) {
    float d = length(uv);
    if (d > 1.0) discard;

    // 8-fold octet pattern (colour octet representation)
    float angle   = atan(uv.y, uv.x) + u_time * 4.0;
    float oct     = pow(abs(sin(angle * 4.0)) * abs(cos(angle * 4.0)), 0.4);
    float ring    = smoothstep(0.06, 0.0, abs(d - 0.55 - 0.1 * sin(u_time * 3.0)));
    float core    = smoothstep(0.2, 0.0, d);

    vec3  col     = mix(vec3(1.0, 0.55, 0.1), vec3(0.9, 0.3, 0.05), oct);
    vec3  emission= col * (ring * 6.0 + core * 8.0) * oct;

    return Surface(col * 0.3, emission, 0.8, 0.2,
                   smoothstep(1.0, 0.8, d) * (oct * 0.6 + ring * 1.5 + core));
}

Surface surface_photon(vec2 uv) {
    float d = length(uv);
    if (d > 1.0) discard;

    // Oscillating EM rings — transverse wave pattern
    float phase    = d * 18.0 - u_time * 7.0;
    float wave     = 0.5 + 0.5 * cos(phase);
    float ring_env = exp(-d * 3.5);
    float core     = smoothstep(0.12, 0.0, d);

    // Polarisation spiral
    float ang      = atan(uv.y, uv.x) + u_time * 2.0;
    float spiral   = 0.5 + 0.5 * sin(ang * 2.0 + d * 10.0);

    vec3 white     = vec3(1.0, 0.98, 0.93);
    vec3 emission  = white * (wave * ring_env * 5.0 + core * 12.0)
                   * mix(1.0, spiral, 0.3);

    return Surface(white * 0.2, emission, 0.0, 1.0,
                   (wave * ring_env + core) * smoothstep(1.0, 0.5, d));
}

Surface surface_dark_matter(vec2 uv) {
    float d = length(uv);
    if (d > 1.0) discard;

    // Nearly invisible — only revealed by gravitational lensing shimmer
    float lensing = noise2(uv * 4.0 + u_time * 0.3) * 0.15;
    float edge    = smoothstep(0.7, 1.0, d) * 0.6;   // dim edge glow only

    vec3 col = vec3(0.22, 0.13, 0.40);
    return Surface(col * 0.05, col * (edge + lensing) * 0.8, 0.0, 0.9,
                   (lensing + edge) * 0.35);
}

Surface surface_string(vec2 uv) {
    float d = length(uv);
    // String is elongated: squash UV
    float d_string = length(vec2(uv.x * 0.25, uv.y));
    if (d_string > 1.0) discard;

    // Standing wave along σ direction (horizontal)
    float sigma    = uv.x;                    // σ ∈ [-1, 1]
    float mode_n   = 1.0 + floor(u_zoom * 4.0);
    float wave     = sin(mode_n * sigma * 3.14159 + u_time * 3.0);
    float amp      = 0.35 * (1.0 - abs(uv.x)) * wave;

    // The displacement is perpendicular to σ
    float vib_d    = abs(uv.y - amp);
    float tube     = smoothstep(0.12, 0.0, vib_d);

    // Colour by mode number: mode 1=violet (graviton), mode 2=blue, etc.
    vec3 mode_col  = mix(vec3(0.6, 0.5, 1.0), vec3(0.9, 0.6, 0.1), (mode_n - 1.0) / 5.0);
    vec3 emission  = mode_col * tube * 6.0;
    // Endpoint glow (D-brane attachment)
    float ep1      = smoothstep(0.15, 0.0, length(uv - vec2(-1.0, amp)));
    float ep2      = smoothstep(0.15, 0.0, length(uv - vec2( 1.0, amp)));
    emission      += vec3(1.0, 0.9, 0.5) * (ep1 + ep2) * 10.0;

    return Surface(mode_col * 0.2, emission, 0.0, 0.3, tube + ep1 + ep2);
}

Surface surface_nucleus(vec2 uv) {
    float d = length(uv);
    if (d > 1.0) discard;

    // Shell model: concentric shells alternating proton/neutron colours
    float shell    = fract(d * 4.0);
    float shell_n  = floor(d * 4.0);
    vec3 proton_c  = vec3(0.95, 0.75, 0.15);
    vec3 neutron_c = vec3(0.45, 0.55, 0.70);
    vec3 shell_col = mix(proton_c, neutron_c, mod(shell_n, 2.0));

    // Nuclear binding ripple
    float ripple   = exp(-shell * 4.0) * sin(shell * 20.0 - u_time * 5.0);
    float core     = smoothstep(0.15, 0.0, d);

    // He-4 is doubly magic (Z=N=2): brighter, more resonant
    float magic    = (v_emission_str > 3.0) ? 2.0 : 1.0;

    vec3  emission = shell_col * (ripple * 2.0 + core * 6.0) * magic;
    float alpha    = smoothstep(1.0, 0.7, d) * (0.7 + 0.3 * abs(ripple));

    return Surface(shell_col * 0.4, emission, 0.3, 0.5, alpha);
}

Surface surface_star(vec2 uv) {
    float d = length(uv);
    if (d > 1.0) discard;

    // Limb darkening: I(μ) = I₀(a + b·μ) where μ = cos(θ) = √(1-r²)
    float mu       = sqrt(max(0.0, 1.0 - d * d));
    float limb     = 0.6 + 0.4 * mu;    // standard solar limb darkening

    // Surface convection: Voronoi-like noise cells
    float conv     = noise2(uv * 8.0 + u_time * 0.1) * 0.3
                   + noise2(uv * 16.0 + u_time * 0.07) * 0.15;

    // Blackbody colour: T_star mapped to colour temperature
    // Pop III stars are ~100,000 K — pure blue-white
    vec3 star_col  = mix(vec3(1.0, 0.95, 0.8), vec3(0.7, 0.85, 1.0),
                         clamp(v_emission_str / 20.0, 0.0, 1.0));

    vec3 emission  = star_col * limb * (8.0 + conv * 4.0) * v_emission_str;
    float alpha    = smoothstep(1.0, 0.85, d);

    return Surface(star_col * 0.3, emission, 0.9, 0.1, alpha);
}

Surface surface_generic(vec2 uv) {
    float d = length(uv);
    if (d > 1.0) discard;
    float core   = smoothstep(0.25, 0.0, d);
    float corona  = smoothstep(1.0, 0.25, d) * (1.0 - core);
    vec3 emission = v_emission * v_emission_str * (core * 4.0 + corona * 1.5);
    return Surface(v_albedo * corona, emission, v_metallic, v_roughness,
                   smoothstep(1.0, 0.75, d) * 0.92);
}

// ── Main ───────────────────────────────────────────────────────
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;    // remap [0,1]→[-1,1]

    Surface s;
    switch(v_type) {
        case 1:  s = surface_quark(uv);       break;
        case 2:  s = surface_gluon(uv);       break;
        case 3:  s = surface_photon(uv);      break;
        case 4:  s = surface_dark_matter(uv); break;
        case 5:  s = surface_string(uv);      break;
        case 6:  s = surface_nucleus(uv);     break;
        case 8:  s = surface_star(uv);        break;
        default: s = surface_generic(uv);     break;
    }

    if (s.alpha < 0.01) discard;

    // Linear depth for SSAO / DoF
    float ndc_depth = gl_FragCoord.z * 2.0 - 1.0;
    float lin_depth = (2.0 * u_near * u_far) / (u_far + u_near - ndc_depth * (u_far - u_near));

    gAlbedoMetallic  = vec4(s.albedo, s.metallic);
    gNormalRoughness = vec4(normalize(v_normal) * 0.5 + 0.5, s.roughness);
    gEmission        = vec4(s.emission, s.alpha);
    gPositionDepth   = vec4(v_world_pos, lin_depth / u_far);
}
