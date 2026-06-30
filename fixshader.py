f=open('shaders/gbuffer.frag',encoding='utf-8').read() 
f=f.replace('float noise2(vec2 p)','vec2 noise2_v(vec2 p)') 
f=f.replace('return fract','return fract') 
open('shaders/gbuffer.frag','w',encoding='utf-8').write(f) 
