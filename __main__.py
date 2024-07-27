import numpy as np
import pygame
import numpy
import taichi as ti

# settings
res = width, height = 1600, 900
offset = numpy.array([1.3 * width, height]) // 2
max_iter = 30
zoom = 2.2 / height

# texture
texture = pygame.image.load('gr.webp')
texture_size = min(texture.get_size()) - 1
texture_array = pygame.surfarray.array3d(texture).astype(dtype=np.uint32)

@ti.data_oriented
class Fractal:
    def __init__(self, aapp):
        self.app = aapp
        self.screen_array = numpy.full((width, height, 3), [0, 0, 0], dtype=numpy.uint32)
        #taichi arch
        ti.init(arch=ti.cuda)
        #taichi fields
        self.screen_field = ti.Vector.field(3, ti.uint32, (width, height))
        self.texture_field = ti.Vector.field(3, ti.uint32, texture.get_size())
        self.texture_field.from_numpy(texture_array)
        #settings
        self.vel = 0.01
        self.zoom, self.scale = 2.2 / height, 0.993
        self.increment = ti.Vector([0.0, 0.0])
        self.max_iter, self.max_iter_limit = 30, 5500
        #d-time
        self.app_speed = 1/4000
        self.prev_time = pygame.time.get_ticks()

    def delta_time(self):
        time_now = pygame.time.get_ticks() - self.prev_time
        self.prev_time = time_now
        return time_now * self.app_speed

    @ti.kernel
    def render(self, max_iter: ti.int32, zoom: ti.float32, dx: ti.float32, dy: ti.float32):
        for x, y in self.screen_field:
            c = ti.Vector([(x - offset[0]) * zoom - dx, (y - offset[1]) * zoom - dy])
            z = ti.Vector([0.0, 0.0])
            num_iter = 0
            for i in range(max_iter):
                z = ti.Vector([(z.x ** 2 - z.y ** 2 + c.x), (2 * z.x * z.y + c.y)])
                if z.dot(z) > 4:
                    break
                num_iter += 1
            col = int(texture_size * num_iter / max_iter)
            self.screen_field[x, y] = self.texture_field[col, col]

    def controls(self):
        pressed_keys = pygame.key.get_pressed()
        dt = self.delta_time()
        if pressed_keys[pygame.K_a]:
            self.increment[0] += self.vel * dt
        if pressed_keys[pygame.K_d]:
            self.increment[0] -= self.vel * dt
        if pressed_keys[pygame.K_w]:
            self.increment[1] += self.vel * dt
        if pressed_keys[pygame.K_s]:
            self.increment[1] -= self.vel * dt

        if pressed_keys[pygame.K_UP] or pressed_keys[pygame.K_DOWN]:
            inv_scale = 2 - self.scale
            if pressed_keys[pygame.K_UP]:
                self.zoom *= self.scale
                self.vel *= self.scale
            if pressed_keys[pygame.K_DOWN]:
                self.zoom *= inv_scale
                self.vel *= inv_scale

        if pressed_keys[pygame.K_LEFT]:
            self.max_iter -= 1
        if pressed_keys[pygame.K_RIGHT]:
            self.max_iter += 1
        self.max_iter = min(max(self.max_iter, 2), self.max_iter_limit)


    def update(self):
        self.controls()
        self.render(self.max_iter, self.zoom, self.increment[0], self.increment[1])
        self.screen_array = self.screen_field.to_numpy()

    def draw(self):
        pygame.surfarray.blit_array(self.app.screen, self.screen_array)

    def run(self):
        self.update()
        self.draw()


class App:
    def __init__(self):
        self.screen = pygame.display.set_mode(res, pygame.SCALED)
        self.clock = pygame.time.Clock()
        self.fractal = Fractal(self)

    def run(self):
        while True:
            self.screen.fill('black')
            self.fractal.run()
            pygame.display.flip()

            [exit() for i in pygame.event.get() if i.type == pygame.QUIT]
            self.clock.tick()
            pygame.display.set_caption(f'fps:{int(self.clock.get_fps())}')


if __name__ == '__main__':
    app = App()
    app.run()
