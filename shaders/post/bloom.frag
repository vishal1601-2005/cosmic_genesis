// shaders/post/bloom.frag
// Dual Kawase Bloom — physically-based light bleed.
//
// Two-pass algorithm (much faster than Gaussian at large radii):
//   Pass 1 (downsample): threshold + 4-tap Kawase downsample
//   Pass 2 (upsample):   4-tap Kawase upsample, additive blend
//
// This produces the correct physically-motivated glow:
// bright emissive objects (particles at n=1 excitation, stellar cores)
// bleed light into surrounding pixels exactly as camera sensors do.
//
// The threshold is set low (0.8) so even dim particles contribute.
// This is critical for the "equations becoming particles" effect —
// the equation text emits at just above threshold, then blooms into
// the particle as the animation progresses.

#version 410 core

in vec2 v_uv;

uniform sampler2D u_hdr_scene;   // HDR input from lighting pass
uniform sampler2D u_bloom_blur;  // previous bloom mip (for upsample pass)
uniform vec2  u_texel;           // 1.0 / resolution
uniform float u_threshold;       // bloom threshold (default 0.8)
uniform float u_strength;        // bloom blend strength
uniform float u_radius;          // sample radius multiplier
uniform int   u_pass;            // 0=threshold+downsample, 1=upsample

out vec4 frag_color;

// Luminance from HDR colour
float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Threshold: soft knee at threshold
vec3 threshold_color(vec3 c) {
    float lum  = luminance(c);
    float knee = u_threshold * 0.5;
    float rq   = clamp(lum - u_threshold + knee, 0.0, 2.0 * knee);
    rq         = (rq * rq) / (4.0 * knee + 0.00001);
    return c * max(rq, lum - u_threshold) / max(lum, 0.00001);
}

// Kawase downsample: 4 samples offset by ±0.5 texel diagonal
vec3 kawase_down(sampler2D tex, vec2 uv, vec2 texel, float offset) {
    vec2 o = texel * offset;
    vec3 s = vec3(0.0);
    s += texture(tex, uv + vec2(-o.x, -o.y)).rgb;
    s += texture(tex, uv + vec2( o.x, -o.y)).rgb;
    s += texture(tex, uv + vec2(-o.x,  o.y)).rgb;
    s += texture(tex, uv + vec2( o.x,  o.y)).rgb;
    return s * 0.25;
}

// Kawase upsample: 8 samples in a ring
vec3 kawase_up(sampler2D tex, vec2 uv, vec2 texel, float radius) {
    vec2 o = texel * radius;
    vec3 s = vec3(0.0);
    // 4 cardinal
    s += texture(tex, uv + vec2( o.x,  0.0)).rgb * 2.0;
    s += texture(tex, uv + vec2(-o.x,  0.0)).rgb * 2.0;
    s += texture(tex, uv + vec2( 0.0,  o.y)).rgb * 2.0;
    s += texture(tex, uv + vec2( 0.0, -o.y)).rgb * 2.0;
    // 4 diagonal (half weight)
    s += texture(tex, uv + vec2( o.x,  o.y)).rgb;
    s += texture(tex, uv + vec2(-o.x,  o.y)).rgb;
    s += texture(tex, uv + vec2( o.x, -o.y)).rgb;
    s += texture(tex, uv + vec2(-o.x, -o.y)).rgb;
    return s / 12.0;
}

void main() {
    if (u_pass == 0) {
        // ── Threshold + downsample ───────────────────────
        vec3 s  = kawase_down(u_hdr_scene, v_uv, u_texel, u_radius);
        vec3 th = threshold_color(s);
        frag_color = vec4(th, 1.0);

    } else {
        // ── Upsample + blend with scene ──────────────────
        vec3 blur  = kawase_up(u_bloom_blur, v_uv, u_texel, u_radius);
        vec3 scene = texture(u_hdr_scene, v_uv).rgb;
        // Additive blend: bloom adds to scene (physically correct — no clamping)
        frag_color = vec4(scene + blur * u_strength, 1.0);
    }
}
