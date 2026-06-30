import os 
for root,dirs,files in os.walk("shaders"): 
    for f in files: 
        p=os.path.join(root,f) 
        c=open(p,encoding="utf-8",errors="ignore").read() 
        if "noise3" in c: print(p,c.count("noise3"),"times") 
