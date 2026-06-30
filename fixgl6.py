lines=open("shaders/gbuffer.frag",encoding="utf-8").readlines() 
for i in range(44,55): print(i+1, repr(lines[i].rstrip())) 
