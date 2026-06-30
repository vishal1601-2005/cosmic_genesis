import os 
data=[] 
for root,dirs,files in os.walk("shaders"): 
 for f in files: 
  p=os.path.join(root,f) 
  lines=open(p,encoding="utf-8",errors="ignore").readlines() 
  n=len(lines) 
  l45=lines[44].strip()[:60] if n else "" 
  data.append((p,l45)) 
for p,l in data: print(p,"|",l) 
