#version 410 core
in vec2 v_uv;
in vec3 v_color;
out vec4 frag_color;
void main(){
    vec2 uv = v_uv * 2.0 - 1.0;
    float d = length(uv);
    if(d > 1.0) discard;
    float alpha = pow(1.0 - d, 1.4);
    vec3 core = mix(v_color, vec3(1.0), (1.0-d)*0.5);
    frag_color = vec4(core, alpha);
}
