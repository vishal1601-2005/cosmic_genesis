#version 410 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
out vec3 v_color;
uniform float u_time;
void main(){
    v_uv = in_uv;
    v_color = vec3(0.8, 0.7, 1.0);
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
