#version 410 core
in vec3 vc;
in vec2 uv;
out vec4 fc;
uniform float u_time;
void main(){float d=length(uv);if(d>1.0)discard;float core=smoothstep(0.3,0.0,d);float glow=smoothstep(1.0,0.0,d);vec3 col=mix(vc,vec3(1.0),core*0.7);fc=vec4(col,glow*0.95);}
