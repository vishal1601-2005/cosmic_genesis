#version 410 core
// forbidden.frag — Renders the visual when a forbidden interaction is attempted.
//
// Shows a radial shockwave ring + red prohibition symbol + the equation text
// is handled by the HUD overlay. This shader draws the background flash effect.
//
// u_center: screen-space position of the attempted interaction
// u_time_since: seconds since the event (animation clock, 0→fade)
// u_reason_type: 0=charge, 1=colour/confinement, 2=energy, 3=epoch, 4=baryon, 5=GSO

in vec2 v_uv;          // fullscreen quad [-1,1]²

uniform vec2  u_center;          // screen UV of event (0..1)
uniform float u_time_since;      // seconds since event
uniform int   u_reason_type;     // which conservation law
uniform float u_aspect;          // window aspect ratio

out vec4 frag_color;

// ── Colour palette for each violation type ─────────────────
vec3 violation_color(int t) {
    if (t == 0) return vec3(1.0, 0.85, 0.1);   // charge → gold
    if (t == 1) return vec3(1.0, 0.2, 0.05);   // confinement → deep red
    if (t == 2) return vec3(0.3, 0.7, 1.0);    // energy → cold blue
    if (t == 3) return vec3(0.6, 0.3, 1.0);    // epoch → purple
    if (t == 4) return vec3(0.1, 1.0, 0.4);    // baryon → green
    if (t == 5) return vec3(1.0, 0.5, 0.9);    // GSO → pink
    return vec3(1.0, 0.3, 0.3);                // generic red
}

void main() {
    if (u_time_since > 1.5) discard;

    float t   = u_time_since;
    float fade= 1.0 - smoothstep(0.6, 1.5, t);
    vec3  col = violation_color(u_reason_type);

    // Screen UV (0..1)
    vec2 screen_uv = v_uv * 0.5 + 0.5;
    // Aspect-corrected distance from event center
    vec2 delta = screen_uv - u_center;
    delta.x   *= u_aspect;
    float d   = length(delta);

    // ── Expanding shockwave ring ─────────────────────────
    float ring_r   = t * 0.55;        // ring expands outward
    float ring_w   = 0.018;
    float ring_mask= smoothstep(ring_w, 0.0, abs(d - ring_r));
    float ring_alpha= ring_mask * fade * (1.0 - t * 0.5);

    // ── Inner radial flash (brief) ───────────────────────
    float flash_r = 0.08;
    float flash   = smoothstep(flash_r, 0.0, d) * exp(-t * 8.0);

    // ── Pulsing radial spokes ─────────────────────────────
    float angle   = atan(delta.y, delta.x);
    int   n_spokes= (u_reason_type == 1) ? 6 : 4;   // colour = 6, others = 4
    float spoke   = abs(sin(float(n_spokes) * angle * 0.5));
    float spoke_a = spoke * smoothstep(0.25, 0.0, abs(d - ring_r * 0.5)) * fade * 0.4;

    float total_alpha = clamp(ring_alpha + flash * 0.8 + spoke_a, 0.0, 0.9);

    // Red "slash" vignette at the center
    float slash1 = smoothstep(0.006, 0.0, abs(delta.x * cos(0.8) - delta.y * sin(0.8)));
    float slash2 = smoothstep(0.006, 0.0, abs(delta.x * cos(-0.8) - delta.y * sin(-0.8)));
    float X_mark = (slash1 + slash2) * smoothstep(0.06, 0.0, d) * exp(-t * 6.0);

    total_alpha = clamp(total_alpha + X_mark * 0.7, 0.0, 0.9);

    frag_color = vec4(col, total_alpha);
}
