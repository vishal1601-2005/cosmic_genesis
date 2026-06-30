lines=open("main.py",encoding="utf-8").readlines() 
out=[] 
skip=False 
for l in lines: 
    if "Draw background nebula" in l: skip=True 
    if skip and "nb+30" in l: skip=False; continue 
    if not skip: out.append(l) 
open("main.py","w",encoding="utf-8").writelines(out) 
print("done") 
