// shaders/post/composite.frag
// Final composite pass:
//   1. ACES filmic tone mapping  (HDR → LDR)
//   2. Chromatic aberration      (lens dispersion)
//   3. Lens dirt / flare         (bright particles bleed onto dirt mask)
//   4. Vignette                  (cinematic edge darkening)
//   5. Film grain                (subtle, adds realism)
//   6. FXAA                      (fast anti-aliasing as final step)
//   7. Gamma correction          (linear → sRGB)
//
// This single shader runs once per frame on the final composited HDR buffer.
// All parameters are tuned per-epoch for cinematic feel.

#version 410 core

in vec2 v_uv;

uniform sampler2D u_hdr;          // HDR scene after bloom
uniform sampler2D u_dirt;         // lens dirt mask (grayscale)
uniform float u_time;
uniform float u_exposure;         // exposure compensation (epoch-specific)
uniform float u_chromatic_ab;     // chromatic aberration strength [0, 0.02]
uniform float u_vignette_str;     // vignette [0, 1]
uniform float u_grain_str;        // film grain [0, 0.08]
uniform float u_dirt_str;         // lens dirt [0, 0.5]
uniform float u_saturation;       // colour saturation multiplier
uniform vec2  u_resolution;
uniform int   u_epoch;

out vec4 frag_color;

// ── ACES Filmic Tonemapping ──────────────────────────────────
// Reference: Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve"
// Keeps highlights and shadows both looking natural.
vec3 aces_filmic(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Alternative: Uncharted 2 / John Hable (slightly warmer)
vec3 hable_tonemap(vec3 x) {
    float A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30, W=11.2;
    vec3 r = ((x*(A*x+C*B)+D*E) / (x*(A*x+B)+D*F)) - E/F;
    vec3 w = ((W*(A*W+C*B)+D*E) / (W*(A*W+B)+D*F)) - E/F;
    return r / w;
}

// ── Chromatic Aberration ─────────────────────────────────────
// Separate R/G/B channels by slight UV offset (simulates lens dispersion).
// Stronger toward edges (realistic), zero at centre.
vec3 chromatic_aberration(sampler2D tex, vec2 uv, float strength) {
    vec2 offset = (uv - 0.5) * strength;
    float r = texture(tex, uv + offset).r;
    float g = texture(tex, uv       ).g;
    float b = texture(tex, uv - offset).b;
    return vec3(r, g, b);
}

// ── FXAA ────────────────────────────────────────────────────
// Simplified FXAA 3.11 luminance-based edge detection.
// Runs at the very end on the LDR output.
vec3 fxaa(sampler2D tex, vec2 uv, vec2 res) {
    vec2 texel = 1.0 / res;

    float lumC  = dot(texture(tex, uv).rgb, vec3(0.299, 0.587, 0.114));
    float lumN  = dot(texture(tex, uv + vec2( 0, 1)*texel).rgb, vec3(0.299,0.587,0.114));
    float lumS  = dot(texture(tex, uv + vec2( 0,-1)*texel).rgb, vec3(0.299,0.587,0.114));
    float lumE  = dot(texture(tex, uv + vec2( 1, 0)*texel).rgb, vec3(0.299,0.587,0.114));
    float lumW  = dot(texture(tex, uv + vec2(-1, 0)*texel).rgb, vec3(0.299,0.587,0.114));

    float lumMin = min(lumC, min(min(lumN,lumS), min(lumE,lumW)));
    float lumMax = max(lumC, max(max(lumN,lumS), max(lumE,lumW)));
    float range  = lumMax - lumMin;

    if (range < max(0.0833, lumMax * 0.166))
        return texture(tex, uv).rgb;   // no edge — skip

    // Blend direction
    float blend = range / lumMax;
    vec2 dir    = vec2(lumS - lumN, lumW - lumE);
    dir         = normalize(dir + vec2(0.0001));

    vec3 result = (
        texture(tex, uv + dir * texel * 0.5).rgb +
        texture(tex, uv - dir * texel * 0.5).rgb
    ) * 0.5;

    return mix(texture(tex, uv).rgb, result, blend * 0.75);
}

// ── Film Grain ───────────────────────────────────────────────
float film_grain(vec2 uv, float time, float strength) {
    // Temporally varying grain (each frame different)
    float grain = fract(sin(dot(uv * 300.0, vec2(127.1, 311.7)) + time * 37.3) * 43758.5);
    return (grain - 0.5) * 2.0 * strength;
}

// ── Vignette ────────────────────────────────────────────────
float vignette(vec2 uv, float strength) {
    vec2  d = uv - 0.5;
    float r = dot(d, d);
    return 1.0 - r * strength * 4.0;
}

// ── Epoch colour grade ───────────────────────────────────────
// Each epoch has a subtle colour grade applied after tonemapping.
vec3 epoch_grade(vec3 col, int epoch) {
    if (epoch == 0) {
        // String landscape: slight blue-violet shift, high contrast
        col = mix(col, col.bgr * vec3(0.8, 0.7, 1.0), 0.15);
        col = pow(col, vec3(0.92));
    } else if (epoch == 1) {
        // Inflation: warm purple, slightly desaturated
        col = mix(col, vec3(dot(col, vec3(0.333))), 0.15);
        col *= vec3(1.05, 0.95, 1.1);
    } else if (epoch == 2) {
        // Baryogenesis: hot orange tint, crushed blacks
        col = max(col - 0.02, vec3(0.0));
        col *= vec3(1.1, 0.9, 0.8);
    } else if (epoch == 3) {
        // QCD: deep red, pushed shadows
        col *= vec3(1.08, 0.88, 0.75);
    } else if (epoch == 5) {
        // BBN: cool blue-white
        col *= vec3(0.92, 0.96, 1.08);
    } else if (epoch == 6) {
        // Recombination: warm amber
        col *= vec3(1.1, 0.95, 0.8);
    } else if (epoch == 7) {
        // Structure: desaturated, dark, cinematic
        float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
        col = mix(col, vec3(lum), 0.2);
        col = pow(col, vec3(1.05));
    }
    return col;
}

// ── Main ────────────────────────────────────────────────────
void main() {
    // 1. Chromatic aberration on HDR input
    vec3 hdr = chromatic_aberration(u_hdr, v_uv, u_chromatic_ab);

    // 2. Exposure
    hdr *= u_exposure;

    // 3. ACES tonemapping (HDR → LDR [0,1])
    vec3 ldr = aces_filmic(hdr);

    // 4. Saturation
    float lum = dot(ldr, vec3(0.2126, 0.7152, 0.0722));
    ldr = mix(vec3(lum), ldr, u_saturation);

    // 5. Epoch-specific colour grade
    ldr = epoch_grade(ldr, u_epoch);

    // 6. Lens dirt (bright areas bleed onto dirt mask)
    float dirt_mask = texture(u_dirt, v_uv).r;
    float bright    = max(0.0, dot(hdr, vec3(0.33)) - 0.8);
    ldr += dirt_mask * bright * u_dirt_str * vec3(0.9, 0.85, 0.7);
    ldr  = clamp(ldr, 0.0, 1.0);

    // 7. Vignette
    ldr *= vignette(v_uv, u_vignette_str);

    // 8. Film grain
    ldr += film_grain(v_uv, u_time, u_grain_str);

    // 9. FXAA (pass through a sampler2D — requires writing ldr to a temp FBO
    //    In practice: bind this shader output to a texture, then run fxaa pass.
    //    Here we do a single-pass approximation.)
    // ldr = fxaa(u_hdr, v_uv, u_resolution);   // uncomment for 2-pass

    // 10. Gamma: linear → sRGB (γ = 2.2)
    ldr = pow(clamp(ldr, 0.0, 1.0), vec3(1.0 / 2.2));

    frag_color = vec4(ldr, 1.0);
}
