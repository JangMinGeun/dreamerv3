import numpy as np
from PIL import Image
from gym import spaces
from gym.utils import seeding
import sys, math, random, tqdm, os, gym, pygame, Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)


# ================================================================
# ========================= DEFINITION ===========================
# ================================================================

# Customized Polygon MUST be in the (1,1) size box.
TRIANGLE_POLY = [
    (0., 0.732051), (-1., -1.), (1., -1.)
]
TRIANGLE_POLY = [(x * 0.95, y*0.95) for (x,y) in TRIANGLE_POLY]

IRON_POLY = [
    (-1., -1.), (1., -1.), (1., 0.), (0.33, 0.), (0.33, 1.), (-0.33, 1.), (-0.33, 0.), (-1.0, 0.)
]
IRON_POLY = [(x * 0.95, y*0.95) for (x,y) in IRON_POLY]

HEXAGON_POLY = [
    (1., 0.), (0.5, 0.866025), (-0.5, 0.866025), (-1., 0.), (-0.5, -0.866025), (0.5, -0.866025)
]
HEXAGON_POLY = [(x * 0.95, y*0.95) for (x,y) in HEXAGON_POLY]

TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 64, 64
PPM = 20.0 / (640 // SCREEN_WIDTH)  # pixels per meter

WALLS = [
    ((SCREEN_WIDTH//(PPM * 2), SCREEN_HEIGHT//PPM), (16,1)), # top
    ((SCREEN_WIDTH//(PPM * 2), 0), (16,1)), # bottom
    ((0, SCREEN_HEIGHT//(PPM * 2)), (1,15)), # left
    ((SCREEN_WIDTH//PPM, SCREEN_HEIGHT//(PPM * 2)), (1,15)) # right
]

# ================================================================
# ========================= DEFINITION ===========================
# ================================================================

discrete_action_x = [-20, 0, 20, -10, 0, 10, -20, -10, 0, 10, 20, -10, 0, 10, -20, 0, 20]
discrete_action_y = [20, 20, 20, 10, 10, 10, 0, 0, 0, 0, 0, -10, -10, -10, -20, -20, -20]

class Elastic2D(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': TARGET_FPS
    }

    def __init__(self, **kwargs):
        self._seed()

        pygame.display.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)  # size, flags, depth
        self.screen.fill((0, 0, 0, 0))
        pygame.display.set_caption('Simple pygame example')
        self.clock = pygame.time.Clock()
        self.world = world(gravity=(0,0))
        
        self.walls = None
        self.objects = None

        self.colors = kwargs['colors']
        self.numofobjs =  kwargs['numofobjs']
        self.objs = kwargs['objs']
        self.sizes = kwargs['sizes']

        self._step = 0
        self._length = 300

        # Defin Draw Functions
        def my_draw_polygon(polygon, body, fixture):
            vertices = [(body.transform * v) * PPM for v in polygon.vertices]
            vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
            pygame.draw.polygon(self.screen, body.userData["color"], vertices)
        polygonShape.draw = my_draw_polygon

        def my_draw_circle(circle, body, fixture):
            position = body.transform * circle.pos * PPM
            position[1] = SCREEN_HEIGHT - position[1]
            pygame.draw.circle(self.screen, body.userData["color"], [int(x) for x in position], int(circle.radius * PPM))
        circleShape.draw = my_draw_circle

        # GYM attributes
        low = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)
        high = np.ones((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8) * 255
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(17)

        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.walls: return

        for wall in self.walls:
            self.world.DestroyBody(wall)
        self.walls = None

        for obj in self.objects:
            self.world.DestroyBody(obj)
        self.objects = None

    def get_random_position_and_angle(self):
        sqrt2 = 1.414215
        angle = random.random() * 3.1415
        size = random.randint(self.sizes[0], self.sizes[1])
        min_x, min_y = int(sqrt2 * size + 2), int(sqrt2 * size + 2) 
        max_x, max_y = int((SCREEN_WIDTH // PPM) - sqrt2 * size - 1), int((SCREEN_HEIGHT // PPM) - sqrt2 * size - 1)
        x_candidate, y_candidate= random.randint(min_x, max_x), random.randint(min_y, max_y)
                    
        return  (x_candidate, y_candidate), angle, size
    
    def reset(self):
        while True: 
            self._destroy()
            
            # create walls for boudary
            self.walls = []
            for pos, size in WALLS:
                self.walls.append(self.world.CreateStaticBody(position=pos, shapes=polygonShape(box=size)))
            for wall in self.walls:
                wall.userData = {}
                wall.userData["color"] = (255, 255, 255, 255)    
                wall.userData["shape"] = 'wall'
            
            # create various objects 
            self.objects = []
            for idx in range(self.numofobjs):
                obj = random.choice(self.objs)
                pos, angle, size = self.get_random_position_and_angle()
                temp_body = self.world.CreateDynamicBody(position=pos, angle=angle)
                temp_body.userData = {}
                temp_body.userData["shape"] = obj
        
                if idx == 0:
                    temp_body.userData["color"] = (255, 0, 0, 255)
                else:
                    temp_body.userData["color"] = random.choices(self.colors)[0]

                if obj == "square":
                    temp_fixture = temp_body.CreatePolygonFixture(box=(size, size), density=1, friction=0.3, restitution=1.0)
                elif obj == "triangle":
                    temp_fixture = temp_body.CreatePolygonFixture(shape=polygonShape(vertices=[(x*size , y*size) for (x, y) in TRIANGLE_POLY]),
                                density=1, friction=0.3, restitution=1.0)
                elif obj == "iron":
                    temp_fixture = temp_body.CreatePolygonFixture(shape=polygonShape(vertices=[(x*size , y*size) for (x, y) in IRON_POLY]),
                                density=1, friction=0.3, restitution=1.0)
                elif obj == "hexagon":
                    temp_fixture = temp_body.CreatePolygonFixture(shape=polygonShape(vertices=[(x*size , y*size) for (x, y) in HEXAGON_POLY]),
                                density=1, friction=0.3, restitution=1.0)
                else:   # circle
                    temp_fixture = temp_body.CreateCircleFixture(radius=size, density=1, friction=0.3, restitution=1.0)
                self.objects.append(temp_body)

            # Ensure all objects are existed in boundary walls 
            object_disappeared = False
            for obj in self.world.bodies:
                if not (obj.userData["shape"] == "wall"):
                    if not (1 < obj.position[0] < ((SCREEN_WIDTH // PPM) - 1) and 1 < obj.position[1] < ((SCREEN_HEIGHT // PPM) - 1)):
                        object_disappeared = True
            if not object_disappeared:  break

        return self.render(mode="rgb_array")

    def step(self, action):
        
        self.objects[0].ApplyForceToCenter([discrete_action_x[int(action)] * 300, discrete_action_y[int(action)] * 300], True)
        
        images = []
        for i in range(300):
            self.world.Step(TIME_STEP, 10, 10)

        pygame.display.flip()
        self.clock.tick(TARGET_FPS)
        
        # Draw the world
        self.screen.fill((0, 0, 0, 0))
        for body in self.world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)

        # Stop all objects
        for body in self.world.bodies:
            body.linearVelocity = (0., 0.)
            body.angularVelocity = 0.

        img = self.render(mode="rgb_array")
        images.append(Image.fromarray(img))

        reward = 0
        done = self._length and self._step>=self._length
        return img, reward, done, {}

    def render(self, mode='human', close=False):
        string_image = pygame.image.tostring(self.screen, 'RGB')
        temp_surf = pygame.image.fromstring(string_image,(SCREEN_WIDTH, SCREEN_HEIGHT),'RGB' )
        img_arr = pygame.surfarray.array3d(temp_surf).swapaxes(0,1)
        return img_arr



