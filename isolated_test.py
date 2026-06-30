import pygame, moderngl
pygame.init()
pygame.display.set_mode((800,600), pygame.OPENGL|pygame.DOUBLEBUF, 24)
ctx = moderngl.create_context()
from render.simple_gpu import SimpleGPURenderer
r = SimpleGPURenderer(ctx, 800, 600)
class P:
    x = 0.0
    y = 0.0
    color_rgb = (1.0, 0.5, 0.8)
    radius = 0.05
clock = pygame.time.Clock()
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            running = False
    r.render([P()], (0.05,0.05,0.1), 0.0)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
