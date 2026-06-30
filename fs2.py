import os
for r,d,files in os.walk("shaders"):
 for f in files:
  p=os.path.join(r,f)
  ls=open(p,encoding="utf-8",errors="ignore").readlines()
  print(f, len(ls), "lines", ls[44].strip()[:50] if len(ls)>44 else "")
