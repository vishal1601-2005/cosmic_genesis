// shaders/worldsheet.frag
// Renders the worldsheet Xᵘ(τ,σ) of a single string as a glowing ribbon.
//
// Each fragment corresponds to a (τ,σ) point on the worldsheet.
// The colour encodes:
//   - Mode amplitude |αₙ|²  → hue (n=1 violet/graviton, n=2 blue, etc.)
//   - Fermionic field |ψ|    → brightness modulation
//   - Mass m²               → saturation (massless=pure, massive=white)
//   - Tachyon (m²<0)        → deep red, pulsing instability

#version 410 core

in float v_mode_amp;     // total mode excitation |αₙ|²
in float v_mass_sq;      // m² = (N-1)/α′
in float v_sigma;        // worldsheet coordinate σ ∈ [0, π]
in float v_psi;          // |ψ₊| fermionic field amplitude
in vec3  v_world_pos;
in float v_excitation;   // normalised [0,1]

uniform float u_time;
uniform float u_zoom;
uniform int   u_theory;  // 0=super, 1=bosonic, 2=het
uniform float u_g_s;

out vec4 frag_color;

// ── Mode colour palette ────────────────────────────────────
// n=1 (massless): violet/purple — graviton, B-field, dilaton
// n=2: blue
// n=3: cyan
// n=4: green
// n=5+: warm (massive, excited)
vec3 mode_color(float mode_amp, float mass_sq) {
    if (mass_sq < -0.01) {
        // Tachyon: deep red, pulsing
        float pulse = 0.5 + 0.5 * sin(u_time * 8.0);
        return mix(vec3(0.8, 0.05, 0.02), vec3(1.0, 0.3, 0.05), pulse);
    }

    // Excitation level → hue
    float exc = clamp(mass_sq / 4.0, 0.0, 1.0);
    vec3 massless_col = vec3(0.55, 0.42, 1.00);  // violet (graviton)
    vec3 n2_col       = vec3(0.18, 0.52, 1.00);  // blue
    vec3 n3_col       = vec3(0.10, 0.85, 0.80);  // cyan
    vec3 massive_col  = vec3(0.95, 0.62, 0.12);  // warm gold

    if (exc < 0.25)  return mix(massless_col, n2_col,    exc / 0.25);
    if (exc < 0.50)  return mix(n2_col,       n3_col,   (exc - 0.25) / 0.25);
    if (exc < 0.75)  return mix(n3_col,       massive_col,(exc - 0.50) / 0.25);
    return massive_col;
}

// ── Endpoint glow (D-brane attachment) ────────────────────
float endpoint_glow(float sigma) {
    float e1 = exp(-sigma * sigma * 12.0);
    float e2 = exp(-(sigma - 3.14159) * (sigma - 3.14159) * 12.0);
    return max(e1, e2);
}

// ── Main ──────────────────────────────────────────────────
void main() {
    // Base colour from mode structure
    vec3  col     = mode_color(v_mode_amp, v_mass_sq);

    // Standing wave pattern along σ
    float wave    = 0.5 + 0.5 * sin(v_sigma * 2.0 + u_time * 2.0);

    // Fermionic modulation: ψ adds shimmer (superstring only)
    float psi_mod = (u_theory == 0) ? (0.7 + 0.3 * abs(v_psi)) : 1.0;

    // Tachyon instability: flicker
    float tachyon_flicker = 1.0;
    if (v_mass_sq < -0.01 && u_theory == 1) {
        tachyon_flicker = 0.4 + 0.6 * abs(sin(u_time * 12.0 + v_sigma * 5.0));
    }

    // Heterotic: E₈ colour structure — 8 transverse modes visible as bands
    float het_bands = 1.0;
    if (u_theory == 2) {
        het_bands = 0.6 + 0.4 * abs(sin(v_sigma * 8.0));
    }

    // Endpoint glow: D-brane attachment points glow white
    float ep       = endpoint_glow(v_sigma);
    vec3  ep_col   = vec3(1.0, 0.95, 0.8) * ep * 3.0;

    // Emission: string worldsheet glows with its mode amplitude
    float glow_str = 1.5 + v_mode_amp * 2.0;
    vec3  emission = col * glow_str * psi_mod * tachyon_flicker * het_bands
                   * (0.7 + 0.3 * wave) + ep_col;

    // Zoom fade: at Planck zoom the worldsheet becomes fully visible
    float alpha    = mix(0.4, 0.92, u_zoom) * tachyon_flicker;

    // String width: thin line at cosmic zoom, thick ribbon at Planck zoom
    // (Fragment discard for off-ribbon pixels handled by geometry clip in .vert)

    frag_color = vec4(emission, alpha);
}
