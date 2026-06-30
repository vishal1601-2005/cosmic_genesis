#version 410 core
in vec2 v_uv;
in vec3 v_albedo;
in vec3 v_emission;
in float v_emission_str;
in float v_age;
flat in int v_type;
layout(location=0) out vec4 gAlbedoMetallic;
layout(location=1) out vec4 gNormalRoughness;
layout(location=2) out vec4 gEmission;
layout(location=3) out vec4 gPositionDepth;
uniform float u_time;
void main(){
vec2 uv=v_uv*2.0-1.0;
float d=length(uv);
if(d>1.0)discard;
float alpha=pow(1.0-d,1.4);
vec3 core=mix(v_albedo,vec3(1.0),(1.0-d)*0.5);
vec3 emission=v_emission*v_emission_str*(1.0-d*0.5);
gAlbedoMetallic=vec4(core,0.0);
gNormalRoughness=vec4(0.5,0.5,1.0,0.5);
gEmission=vec4(emission,alpha);
gPositionDepth=vec4(0.0,0.0,0.0,0.5);
}
