// shaders/gbuffer.vert
// G-Buffer pass: writes position, normal, albedo, emission, metallic/roughness
// into MRT (Multiple Render Targets) for deferred shading.
//
// Each particle / worldsheet vertex goes through this pass first.
// The lighting pass then reads the G-Buffer textures.

#version 410 core

// Per-vertex
layout(location = 0) in vec3  in_position;
layout(location = 1) in vec3  in_normal;
layout(location = 2) in vec2  in_uv;

// Per-instance (one per particle — instanced draw)
layout(location = 3) in vec3  inst_world_pos;     // particle centre (world)
layout(location = 4) in vec3  inst_color;          // base albedo
layout(location = 5) in vec3  inst_emission;       // emission colour
layout(location = 6) in float inst_emission_str;   // emission strength (HDR > 1)
layout(location = 7) in float inst_radius;         // particle radius
layout(location = 8) in float inst_metallic;       // 0=dielectric, 1=metallic
layout(location = 9) in float inst_roughness;      // 0=mirror, 1=diffuse
layout(location=10) in float  inst_age;            // seconds alive
layout(location=11) in int    inst_type;           // particle type

uniform mat4  u_view;
uniform mat4  u_proj;
uniform vec3  u_cam_right;
uniform vec3  u_cam_up;
uniform float u_time;

out vec3  v_world_pos;
out vec3  v_normal;
out vec2  v_uv;
out vec3  v_albedo;
out vec3  v_emission;
out float v_emission_str;
out float v_metallic;
out float v_roughness;
out float v_age;
flat out int v_type;

void main() {
    // Billboard: face camera regardless of rotation
    // Scale by radius + subtle heartbeat pulse (particles "breathe")
    float pulse = 1.0 + 0.04 * sin(u_time * 3.14159 * 2.0 * 0.8 + inst_world_pos.x * 7.3);
    float r = inst_radius * pulse;

    // Age fade-in: new particles scale from 0
    float age_scale = min(1.0, inst_age * 8.0);
    r *= age_scale;

    vec3 world_pos = inst_world_pos
                   + u_cam_right * in_position.x * r
                   + u_cam_up    * in_position.y * r;

    gl_Position  = u_proj * u_view * vec4(world_pos, 1.0);

    v_world_pos    = world_pos;
    v_normal       = normalize(u_cam_right * in_position.x + u_cam_up * in_position.y
                               + cross(u_cam_right, u_cam_up) * sqrt(max(0.0,
                                 1.0 - dot(in_position.xy, in_position.xy))));
    v_uv           = in_uv;
    v_albedo       = inst_color;
    v_emission     = inst_emission;
    v_emission_str = inst_emission_str;
    v_metallic     = inst_metallic;
    v_roughness    = inst_roughness;
    v_age          = inst_age;
    v_type         = inst_type;
}
