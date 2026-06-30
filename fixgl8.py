f=open("shaders/gbuffer.frag",encoding="utf-8").read() 
n3="float noise3(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}\n" 
f=f.replace("// __ Type-specific",n3+"// __ Type-specific") 
f=f.replace("// \u2500\u2500 Type-specific",n3+"// \u2500\u2500 Type-specific") 
open("shaders/gbuffer.frag","w",encoding="utf-8").write(f) 
print("done") 
