content=open("main.py",encoding="utf-8").read() 
old="    renderer = None" 
new="    renderer = None\n    simple_gpu = None" 
content=content.replace(old,new) 
old2="            use_gpu = False" 
content=content.replace(old2,new2) 
open("main.py","w",encoding="utf-8").write(content) 
