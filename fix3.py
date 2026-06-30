lines=open('main.py',encoding='utf-8').readlines() 
lines[371]='    print("clicked")\n' 
open('main.py','w',encoding='utf-8').writelines(lines) 
print('done') 
