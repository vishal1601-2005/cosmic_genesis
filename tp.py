import pygame,moderngl,numpy as np
pygame.init()
pygame.display.set_mode((800,600),pygame.OPENGL|pygame.DOUBLEBUF,24)
ctx=moderngl.create_context()
prog=ctx.program(vertex_shader=open("pv.glsl").read(),fragment_shader=open("pf.glsl").read())
print("Particle shader OK on",ctx.info["GL_RENDERER"])
pygame.quit()
