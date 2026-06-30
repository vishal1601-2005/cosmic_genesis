#version 410 core
in vec2 in_vert;
in vec3 in_color;
in vec2 in_center;
in float in_radius;
out vec3 vc;
out vec2 uv;
void main(){uv=in_vert;vc=in_color;gl_Position=vec4(in_center+in_vert*in_radius,0,1);}
