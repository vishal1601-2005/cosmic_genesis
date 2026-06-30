lines=open("shaders/gbuffer.frag",encoding="utf-8").readlines() 
lines[43]=lines[43].replace("vec2 noise2_v","float noise2x") 
open("shaders/gbuffer.frag","w",encoding="utf-8").writelines(lines) 
print("done") 
