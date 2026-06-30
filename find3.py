f=open("shaders/gbuffer.frag",encoding="utf-8").read() 
lines=f.split("\n") 
for i,l in enumerate(lines): 
    if "noise3" in l: print(i+1,l[:80]) 
