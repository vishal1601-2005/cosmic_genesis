content=open('main.py',encoding='utf-8').read() 
content=content.replace('pygame.draw.circle(surf,(nb,nb,nb+30),(nx,ny),nr) None','pygame.draw.circle(surf,(nb,nb,nb+30),(nx,ny),nr)') 
open('main.py','w',encoding='utf-8').write(content) 
