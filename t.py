import pygame,moderngl
pygame.init()
pygame.display.set_mode((800,600),pygame.OPENGL|pygame.DOUBLEBUF,24)
ctx=moderngl.create_context()
prog=ctx.program(vertex_shader=open("v.glsl").read(),fragment_shader=open("f.glsl").read())
print("OK",ctx.info["GL_RENDERER"])
pygame.quit()
