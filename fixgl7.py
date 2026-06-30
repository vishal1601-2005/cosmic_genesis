lines=open("shaders/gbuffer.frag",encoding="utf-8").readlines() 
for i,l in enumerate(lines): 
    if "float noise" in l or "vec2 noise" in l or "vec3 noise" in l: print(i+1,l.rstrip()) 
