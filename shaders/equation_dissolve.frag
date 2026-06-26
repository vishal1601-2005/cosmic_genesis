// shaders/equation_dissolve.frag
// THE SIGNATURE VISUAL:
// A physics equation written in glowing text slowly dissolves and
// condenses into a particle. This shader handles both directions:
//   Mode 0 (appear):   equation materialises from void → glowing text
//   Mode 1 (dissolve): text breaks apart → particle forms at centre
//   Mode 2 (particle): full particle (hands off to particle shader)
//
// The dissolve uses a threshold mask driven by a noise field:
// pixels whose noise value < threshold dissolve first (outer chars first).
// The equation text is provided as a texture atlas.
//
// Used for:
//   - Particle discovery: "m² = (n-1)/α′" → graviton
//   - Fusion: "p + n → D + γ" → deuterium nucleus flash
//   - Forbidden: equation appears in red → shatters outward

#version 410 core

in vec2 v_uv;

uniform sampler2D u_eq_tex;       // equation text rendered to texture
uniform sampler2D u_noise_tex;    // 2D blue noise mask
uniform float u_progress;         // 0.0 = full text, 1.0 = full particle
uniform vec3  u_eq_color;         // base equation text colour
uniform vec3  u_particle_color;   // destination particle colour
uniform float u_glow_str;         // glow strength
uniform float u_time;
uniform int   u_mode;             // 0=appear, 1=dissolve, 2=forbidden
uniform vec2  u_center;           // particle centre in UV space
uniform float u_forbidden_radius; // for forbidden: expanding ring

out vec4 frag_color;

// ── Utility ───────────────────────────────────────────────────
float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Blue-noise threshold mask
float dissolve_threshold(vec2 uv, float progress) {
    float noise = texture(u_noise_tex, uv * 3.0 + u_time * 0.01).r;
    return step(noise, progress);
}

// Soft glow around text edges
vec3 text_glow(sampler2D tex, vec2 uv, vec2 texel, float radius, vec3 color) {
    float glow = 0.0;
    int steps = 8;
    for (int i = 0; i < steps; i++) {
        float angle = float(i) / float(steps) * 6.28318;
        vec2  offset = vec2(cos(angle), sin(angle)) * texel * radius;
        glow += texture(tex, uv + offset).a;
    }
    glow /= float(steps);
    return color * glow * u_glow_str;
}

// ── Main ─────────────────────────────────────────────────────
void main() {
    vec2 texel = 1.0 / textureSize(u_eq_tex, 0);

    // Sample equation texture
    vec4 eq   = texture(u_eq_tex, v_uv);
    float text = eq.a;    // text alpha (glyph mask)

    if (u_mode == 0) {
        // ── APPEAR: equation materialises ────────────────
        // Pixels appear in order of noise value, from centre outward
        float dist_center = length(v_uv - 0.5);
        float noise       = texture(u_noise_tex, v_uv * 2.0).r;
        // Appear threshold: inner pixels first
        float threshold   = (1.0 - u_progress) + dist_center * 0.4;
        float visible     = step(noise, 1.0 - threshold);

        vec3 color = u_eq_color;
        // Electric materialisation edge
        float edge = smoothstep(threshold, threshold + 0.05, noise);
        color     += vec3(0.3, 0.6, 1.0) * edge * 3.0;   // blue electric edge

        // Glow
        vec3 glow  = text_glow(u_eq_tex, v_uv, texel, 3.0, u_eq_color);

        frag_color = vec4(color + glow, text * visible);

    } else if (u_mode == 1) {
        // ── DISSOLVE: text → particle ────────────────────
        // Pixels dissolve outward as u_progress → 1
        float noise     = texture(u_noise_tex, v_uv * 2.5 + u_time * 0.05).r;
        float dissolve  = dissolve_threshold(v_uv, u_progress);

        // Dissolving edge glows brightly (like burning paper)
        float edge_mask = smoothstep(u_progress - 0.06, u_progress, noise) * text;
        vec3  edge_glow = mix(u_eq_color * 2.0,
                              u_particle_color * 4.0, u_progress) * edge_mask;

        // Particles streaming toward centre
        vec2  to_centre  = normalize(u_center - v_uv + vec2(0.001));
        float stream     = max(0.0, dot(to_centre,
                              normalize(vec2(sin(u_time*3.0), cos(u_time*3.0)))));
        vec3  streak     = u_particle_color * stream * u_progress * 0.6 * (1.0 - text);

        // Remaining text (not yet dissolved)
        vec3  remain     = u_eq_color * text * (1.0 - dissolve);

        // Core particle forming at centre
        float core_d     = length(v_uv - u_center);
        float core       = smoothstep(0.12 * (1.0 - u_progress * 0.5), 0.0, core_d)
                         * u_progress;
        vec3  core_col   = mix(u_eq_color, u_particle_color, u_progress)
                         * core * 5.0;

        vec3  color      = remain + edge_glow + streak + core_col;
        float alpha      = max(max(text * (1.0 - dissolve), edge_mask),
                               max(core, length(streak) > 0.01 ? 0.4 : 0.0));

        frag_color = vec4(color, alpha);

    } else {
        // ── FORBIDDEN: equation appears in red → shatters ─
        float progress  = u_progress;

        // Text renders in red
        vec3  red_eq    = vec3(1.0, 0.15, 0.1) * text;

        // After 50% progress, shatter outward
        if (progress > 0.5) {
            float shatter   = (progress - 0.5) * 2.0;
            float noise     = texture(u_noise_tex, v_uv).r;
            // Shatter: pixels fly outward from centre
            vec2  from_cen  = v_uv - u_center;
            float shard     = step(1.0 - shatter, noise) * text;
            // Flying shards
            vec2  shard_uv  = v_uv - from_cen * shatter * 0.3 * noise;
            vec4  shard_eq  = texture(u_eq_tex, shard_uv);
            red_eq          = mix(red_eq, vec3(1.0, 0.1, 0.05) * shard_eq.a,
                                  shatter * 0.8);
            // Expanding ring at centre
            float ring_r    = u_forbidden_radius;
            float ring      = smoothstep(0.02, 0.0, abs(length(from_cen) - ring_r));
            red_eq         += vec3(1.0, 0.2, 0.05) * ring * (1.0 - shatter);
        }

        // ⊗ symbol at centre (drawn by CPU/HUD, this adds the glow)
        float symbol_d = length(v_uv - u_center);
        float symbol_g = smoothstep(0.08, 0.0, symbol_d) * (1.0 - progress);
        red_eq        += vec3(1.0, 0.1, 0.05) * symbol_g * 3.0;

        frag_color = vec4(red_eq,
                          max(text * (1.0 - progress), symbol_g));
    }
}
