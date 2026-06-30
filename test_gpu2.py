import pygame,moderngl
pygame.init()
pygame.display.set_mode((800,600),pygame.OPENGL|pygame.DOUBLEBUF,24)
ctx=moderngl.create_context()
v="#version 410\nin vec2 p;\nout vec2 u;\nvoid main(){u=p;gl_Position=vec4(p,0,1);}"
fr="#version 410\nin vec2 u;\nout vec4 c;\nvoid main(){float d=length(u*2-1);if(d;c=vec4(1-d,0.5,1,1);}"
prog=ctx.program(vertex_shader=v,fragment_shader=fr)
print("Shader compiled OK on",ctx.info["GL_RENDERER"])
pygame.quit()
