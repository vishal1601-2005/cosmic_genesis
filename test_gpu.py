import pygame,moderngl
pygame.init()
s=pygame.display.set_mode((800,600),pygame.OPENGL|pygame.DOUBLEBUF,24)
ctx=moderngl.create_context()
print(ctx.info["GL_RENDERER"])
pygame.quit()
