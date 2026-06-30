import moderngl
import numpy as np

class SimpleGPURenderer:
    def __init__(self, ctx, W, H):
        self.ctx = ctx
        self.W = W
        self.H = H
        vert = chr(10).join(['#version 410 core', 'in vec2 in_vert;', 'in vec3 in_color;', 'in vec2 in_center;', 'in float in_radius;', 'in float in_type;', 'out vec3 vc;', 'out vec2 uv;', 'out float vtype;', 'void main(){', 'uv=in_vert;', 'vc=in_color;', 'vtype=in_type;', 'gl_Position=vec4(in_center+in_vert*in_radius,0.0,1.0);', '}'])
        frag_a = chr(10).join(['#version 410 core', 'in vec3 vc;', 'in vec2 uv;', 'in float vtype;', 'out vec4 fc;', 'void main(){', 'float d=length(uv);', 'if (d > 1.0) discard;', 'vec3 col;', 'float alpha;'])
        frag_b = chr(10).join(['if (vtype > 6.5) {', 'float core=smoothstep(0.55,0.0,d);', 'float ray=pow(max(0.0,1.0-d),8.0);', 'col=mix(vc,vec3(1.0),core*0.9+ray*0.3);', 'alpha=smoothstep(1.0,0.2,d);', '} else if (vtype > 5.5) {', 'float fall=smoothstep(1.0,0.0,d);', 'col=vc*0.6;', 'alpha=fall*fall*0.35;', '} else {'])
        frag_c = chr(10).join(['float core=smoothstep(0.25,0.0,d);', 'float rim=smoothstep(0.75,0.55,d)*smoothstep(0.4,0.55,d);', 'col=mix(vc,vec3(1.0),core*0.6);', 'col=col+vec3(1.0,1.0,0.9)*rim*0.5;', 'alpha=smoothstep(1.0,0.4,d)*0.95;', '}', 'fc=vec4(col,alpha);', '}'])
        frag = frag_a + chr(10) + frag_b + chr(10) + frag_c
        self.prog = ctx.program(vertex_shader=vert, fragment_shader=frag)
        quad = np.array([-1,-1,1,-1,-1,1,1,-1,1,1,-1,1], dtype=np.float32)
        self.qvbo = ctx.buffer(quad.tobytes())
        self.ivbo = ctx.buffer(reserve=600000)
        self.vao = ctx.vertex_array(self.prog, [(self.qvbo,"2f","in_vert"),(self.ivbo,"3f 2f 1f 1f /i","in_color","in_center","in_radius","in_type")])
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        print("SimpleGPURenderer ready on", ctx.info["GL_RENDERER"])

    def render(self, particles, bg, t):
        self.ctx.clear(*bg, 1.0)
        if not particles:
            return
        mx = max((abs(getattr(p,"x",1)) for p in particles), default=1.0)
        sc = max(1.0, mx)
        data = []
        for p in particles:
            r2,g2,b2 = getattr(p,"color_rgb",(0.8,0.7,1.0))
            x = getattr(p,"x",0) / sc
            y = getattr(p,"y",0) / sc * 0.8
            tname = type(p).__name__
            if tname == "DarkMatterHalo":
                rad = min(0.30, max(0.06, getattr(p,"radius",0.05)*1.5))
                ptype = 6.0
            elif tname == "Star":
                rad = min(0.06, max(0.015, getattr(p,"radius",0.02)*1.5))
                ptype = 7.0
            else:
                rad = min(0.035, max(0.006, getattr(p,"radius",0.02)*0.8))
                ptype = 0.0
            data += [r2,g2,b2,x,y,rad,ptype]
        arr = np.array(data, dtype=np.float32)
        self.ivbo.write(arr.tobytes())
        self.vao.render(moderngl.TRIANGLES, vertices=6, instances=len(particles))
