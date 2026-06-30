content=open("main.py",encoding="utf-8").read() 
content=content.replace("[renderer] init failed","[renderer skipped]") 
open("main.py","w",encoding="utf-8").write(content) 
