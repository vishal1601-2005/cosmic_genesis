import sys 
content=open("main.py",encoding="utf-8").read() 
old="        if use_gpu and ctx and renderer and pbuf:" 
new="        if False and use_gpu and ctx and renderer and pbuf:" 
content=content.replace(old,new) 
open("main.py","w",encoding="utf-8").write(content) 
print("done") 
