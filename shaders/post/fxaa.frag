// shaders/post/fxaa.frag
// FXAA 3.11 — Fast Approximate Anti-Aliasing.
// Runs as the very last pass on the LDR (post-tonemap) output.
// Input: u_scene (RGB, already gamma-corrected LDR)
// Output: anti-aliased RGB
//
// Algorithm:
//   1. Compute local luminance contrast
//   2. If contrast < threshold: output scene pixel unchanged
//   3. Else: detect edge direction, blend 1–2 pixels along edge
//
// Quality preset:
//   FXAA_QUALITY__PRESET 12 (medium — good for 60fps real-time)

#version 410 core

in vec2 v_uv;

uniform sampler2D u_scene;
uniform vec2      u_rcp_frame;   // 1.0 / resolution (x and y)

out vec4 frag_color;

// Luminance from RGB (perceptual weights)
float luma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

void main() {
    // Sample centre and cardinal neighbours
    vec3  rgbC  = texture(u_scene, v_uv).rgb;
    vec3  rgbN  = textureOffset(u_scene, v_uv, ivec2( 0,  1)).rgb;
    vec3  rgbS  = textureOffset(u_scene, v_uv, ivec2( 0, -1)).rgb;
    vec3  rgbE  = textureOffset(u_scene, v_uv, ivec2( 1,  0)).rgb;
    vec3  rgbW  = textureOffset(u_scene, v_uv, ivec2(-1,  0)).rgb;

    float lumC  = luma(rgbC);
    float lumN  = luma(rgbN);
    float lumS  = luma(rgbS);
    float lumE  = luma(rgbE);
    float lumW  = luma(rgbW);

    float lumMin = min(lumC, min(min(lumN, lumS), min(lumE, lumW)));
    float lumMax = max(lumC, max(max(lumN, lumS), max(lumE, lumW)));
    float range  = lumMax - lumMin;

    // Early exit: no edge here
    const float EDGE_THRESHOLD      = 0.0833;
    const float EDGE_THRESHOLD_MIN  = 0.0312;
    if (range < max(EDGE_THRESHOLD_MIN, lumMax * EDGE_THRESHOLD)) {
        frag_color = vec4(rgbC, 1.0);
        return;
    }

    // Sample diagonal neighbours
    vec3  rgbNW = textureOffset(u_scene, v_uv, ivec2(-1,  1)).rgb;
    vec3  rgbNE = textureOffset(u_scene, v_uv, ivec2( 1,  1)).rgb;
    vec3  rgbSW = textureOffset(u_scene, v_uv, ivec2(-1, -1)).rgb;
    vec3  rgbSE = textureOffset(u_scene, v_uv, ivec2( 1, -1)).rgb;

    float lumNW = luma(rgbNW);
    float lumNE = luma(rgbNE);
    float lumSW = luma(rgbSW);
    float lumSE = luma(rgbSE);

    // Sub-pixel aliasing filter
    float lumL  = (lumN + lumS + lumE + lumW) * 0.25;
    float lumL2 = (lumNW + lumNE + lumSW + lumSE) * 0.25;
    float subpix = abs(lumL - lumC) / range;
    subpix = clamp(subpix * subpix * 0.75, 0.0, 1.0);

    // Edge direction: horizontal vs vertical
    float edgeH = abs(-2.0*lumW + lumNW + lumSW) +
                  abs(-2.0*lumC + lumN  + lumS ) * 2.0 +
                  abs(-2.0*lumE + lumNE + lumSE);
    float edgeV = abs(-2.0*lumN + lumNW + lumNE) +
                  abs(-2.0*lumC + lumW  + lumE ) * 2.0 +
                  abs(-2.0*lumS + lumSW + lumSE);

    bool  isHorizontal = (edgeH >= edgeV);

    // Gradient toward the two neighbours perpendicular to edge
    float lum1  = isHorizontal ? lumN : lumE;
    float lum2  = isHorizontal ? lumS : lumW;
    float grad1 = abs(lum1 - lumC);
    float grad2 = abs(lum2 - lumC);
    bool  steep = (grad1 >= grad2);

    vec2  stepDir = isHorizontal
                  ? vec2(0.0, u_rcp_frame.y)
                  : vec2(u_rcp_frame.x, 0.0);
    if (!steep) stepDir = -stepDir;

    // Blend along the edge
    vec2  uv1   = v_uv + stepDir * 0.5;
    float lumEdge = (lumC + (steep ? lum1 : lum2)) * 0.5;

    // Iterative search along edge (8 iterations, quality preset 12)
    const float STEPS[8] = float[](1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 4.0);
    vec2  edgeDir = isHorizontal
                  ? vec2(u_rcp_frame.x, 0.0)
                  : vec2(0.0, u_rcp_frame.y);

    vec2  uvP = uv1 + edgeDir;
    vec2  uvN = uv1 - edgeDir;
    float lumEndP = luma(texture(u_scene, uvP).rgb) - lumEdge;
    float lumEndN = luma(texture(u_scene, uvN).rgb) - lumEdge;
    bool  doneP   = abs(lumEndP) >= 0.25 * range;
    bool  doneN   = abs(lumEndN) >= 0.25 * range;

    for (int i = 0; i < 8 && !(doneP && doneN); i++) {
        float step = STEPS[i];
        if (!doneP) { uvP += edgeDir * step; lumEndP = luma(texture(u_scene, uvP).rgb) - lumEdge; doneP = abs(lumEndP) >= 0.25 * range; }
        if (!doneN) { uvN -= edgeDir * step; lumEndN = luma(texture(u_scene, uvN).rgb) - lumEdge; doneN = abs(lumEndN) >= 0.25 * range; }
    }

    // Pixel offset
    float distP   = isHorizontal ? abs(uvP.x - v_uv.x) : abs(uvP.y - v_uv.y);
    float distN   = isHorizontal ? abs(uvN.x - v_uv.x) : abs(uvN.y - v_uv.y);
    bool  nearerP = distP < distN;
    float lumEnd  = nearerP ? lumEndP : lumEndN;

    if (((lumC - lumEdge) < 0.0) == (lumEnd < 0.0)) {
        frag_color = vec4(rgbC, 1.0);
        return;
    }

    float dist = nearerP ? distP : distN;
    float total = distP + distN + 1e-4;
    float pixelOffset = 0.5 - dist / total;
    pixelOffset = max(pixelOffset, 0.0);

    // Subpixel offset
    float finalOffset = max(pixelOffset, subpix * 0.75);

    vec2  finalUV  = v_uv + stepDir * finalOffset;
    vec3  rgbFinal = texture(u_scene, finalUV).rgb;

    frag_color = vec4(rgbFinal, 1.0);
}
