ls=open("shaders/gbuffer.vert",encoding="utf-8").readlines()
for i in range(40,52): print(i+1,ls[i].rstrip()[:70] if len(ls)>i else "")
