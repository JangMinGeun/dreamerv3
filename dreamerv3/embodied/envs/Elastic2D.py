import logging
import threading
import embodied
import numpy as np


class Elastic2D(embodied.Env):

  def __init__(
      self,
      repeat=1,
      size=(64, 64),
      **kwargs
  ):
    self._repeat = repeat
    self._size = size

    self.colors = [(0, 255, 0, 255), (0, 0, 255, 255)]
    self.numofobjs =  kwargs['numofobjs']
    self.objs = kwargs['objs']
    self.sizes = kwargs['sizes']

    # Make env
    import gym
    import elastic2d
    self._gymenv = gym.make('elastic2d-v1', colors=self.colors, objs=self.objs, numofobjs=self.numofobjs, sizes=self.sizes)

    from . import from_gym
    self._env = from_gym.FromGym(self._gymenv, obs_key='image')

    # Observations
    self._step = 0
    self._obs_space = self.obs_space
    self._done = True
    self._length = 300

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, 17),
        'reset': embodied.Space(bool),
  }

  def step(self, action):
    # print("step:", self._step)
    print('action', action)
    if action['reset'] or self._done:
        obs = self._reset()
        self._done = False
        return obs

    obs = self._env.step(action)
    self._done = self._length and self._step >= self._length
    self._step += 1
    # print("step:", self._step)
    return obs



  def _reset(self):
    obs = self._env.step({'reset': True})
    self._step = 0
    return obs


  def _obs(self, img, reward=0.0, is_first=False, is_last=False, is_terminal=False):
    obs = {
        'image': img,
        'reward': reward,
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal
    }
    for key, value in obs.items():
        space = self._obs_space[key]
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        assert value in space, (key, value, value.dtype, value.shape, space)
    return obs


  def render(self):
    return self._env.render()

# import embodied
# import numpy as np
# from PIL import Image
# from gym import spaces
# from gym.utils import seeding
# import sys, math, random, tqdm, os, gym, pygame, Box2D
# from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
# import elastic2d
#
#
# # ================================================================
# # ========================= DEFINITION ===========================
# # ================================================================
#
# # Customized Polygon MUST be in the (1,1) size box.
# TRIANGLE_POLY = [
#     (0., 0.732051), (-1., -1.), (1., -1.)
# ]
# TRIANGLE_POLY = [(x * 0.95, y*0.95) for (x,y) in TRIANGLE_POLY]
#
# IRON_POLY = [
#     (-1., -1.), (1., -1.), (1., 0.), (0.33, 0.), (0.33, 1.), (-0.33, 1.), (-0.33, 0.), (-1.0, 0.)
# ]
# IRON_POLY = [(x * 0.95, y*0.95) for (x,y) in IRON_POLY]
#
# HEXAGON_POLY = [
#     (1., 0.), (0.5, 0.866025), (-0.5, 0.866025), (-1., 0.), (-0.5, -0.866025), (0.5, -0.866025)
# ]
# HEXAGON_POLY = [(x * 0.95, y*0.95) for (x,y) in HEXAGON_POLY]
#
# TARGET_FPS = 60
# TIME_STEP = 1.0 / TARGET_FPS
# SCREEN_WIDTH, SCREEN_HEIGHT = 64, 64
# PPM = 20.0 / (640 // SCREEN_WIDTH)  # pixels per meter
#
# WALLS = [
#     ((SCREEN_WIDTH//(PPM * 2), SCREEN_HEIGHT//PPM), (16,1)), # top
#     ((SCREEN_WIDTH//(PPM * 2), 0), (16,1)), # bottom
#     ((0, SCREEN_HEIGHT//(PPM * 2)), (1,15)), # left
#     ((SCREEN_WIDTH//PPM, SCREEN_HEIGHT//(PPM * 2)), (1,15)) # right
# ]
#
# # ================================================================
# # ========================= DEFINITION ===========================
# # ================================================================
#
# discrete_action_x = [-20, 0, 20, -10, 0, 10, -20, -10, 0, 10, 20, -10, 0, 10, -20, 0, 20]
# discrete_action_y = [20, 20, 20, 10, 10, 10, 0, 0, 0, 0, 0, -10, -10, -10, -20, -20, -20]
#
# class Elastic2D(embodied.Env):
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': TARGET_FPS
#     }
#
#     def __init__(self,task, **kwargs):
#         self._seed()
#
#         pygame.display.init()
#         self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)  # size, flags, depth
#         self.screen.fill((0, 0, 0, 0))
#         pygame.display.set_caption('Simple pygame example')
#         self.clock = pygame.time.Clock()
#         self.world = world(gravity=(0,0))
#
#         self.walls = None
#         self.objects = None
#
#         self.colors = [( 0, 255, 0, 255 ), ( 0, 0, 255, 255 )]
#         self.numofobjs =  kwargs['numofobjs']
#         self.objs = kwargs['objs']
#         self.sizes = kwargs['sizes']
#
#         self._repeat = 1
#         self._size = (64, 64)
#         env = gym.make('elastic2d-v1', colors=self.colors, objs=self.objs, numofobjs=self.numofobjs, sizes=self.sizes)
#         from . import from_gym
#         self._env = from_gym.FromGym(env, obs_key='image')
#
#         # Observations
#         self._step = 0
#         self._length = 300
#
#
#         # Defin Draw Functions
#         def my_draw_polygon(polygon, body, fixture):
#             vertices = [(body.transform * v) * PPM for v in polygon.vertices]
#             vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
#             pygame.draw.polygon(self.screen, body.userData["color"], vertices)
#         polygonShape.draw = my_draw_polygon
#
#         def my_draw_circle(circle, body, fixture):
#             position = body.transform * circle.pos * PPM
#             position[1] = SCREEN_HEIGHT - position[1]
#             pygame.draw.circle(self.screen, body.userData["color"], [int(x) for x in position], int(circle.radius * PPM))
#         circleShape.draw = my_draw_circle
#
#         # GYM attributes
#         high = np.array([np.inf] * 8)
#         self.observation_space = spaces.Box(-high, high)
#         self._obs_space = self.obs_space
#         self._done = True
#         self.action_space = spaces.Discrete(17)
#
#         self.reset()
#
#     def _seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def _destroy(self):
#         if not self.walls: return
#
#         for wall in self.walls:
#             self.world.DestroyBody(wall)
#         self.walls = None
#
#         for obj in self.objects:
#             self.world.DestroyBody(obj)
#         self.objects = None
#
#     def get_random_position_and_angle(self):
#         sqrt2 = 1.414215
#         angle = random.random() * 3.1415
#         size = random.randint(self.sizes[0], self.sizes[1])
#         min_x, min_y = int(sqrt2 * size + 2), int(sqrt2 * size + 2)
#         max_x, max_y = int((SCREEN_WIDTH // PPM) - sqrt2 * size - 1), int((SCREEN_HEIGHT // PPM) - sqrt2 * size - 1)
#         x_candidate, y_candidate= random.randint(min_x, max_x), random.randint(min_y, max_y)
#
#         return  (x_candidate, y_candidate), angle, size
#
#     @property
#     def obs_space(self):
#         print('obs_space:', self.observation_space.shape)
#         spaces = {
#             'image': embodied.Space(np.uint8, (64,64,3)),
#             'reward': embodied.Space(np.float32),
#             'is_first': embodied.Space(bool),
#             'is_last': embodied.Space(bool),
#             'is_terminal': embodied.Space(bool),
#         }
#         # spaces.update({
#         #     f'log_achievement_{k}': embodied.Space(np.int32)
#         #     for k in self._achievements})
#         return spaces
#     @property
#     def act_space(self):
#         return {
#             'action': embodied.Space(np.int32, (), 0, 17),
#             'reset': embodied.Space(bool),
#         }
#
#     def _reset(self):
#         obs, _, _, _ = self._env.reset()
#         self._step = 0
#         return obs
#
#     def _obs(self, img, reward = 0.0, is_first=False, is_last = False, is_terminal=False):
#         obs = {
#             'image':img,
#             'reward':reward,
#             "is_first": is_first,
#             "is_last": is_last,
#             "is_terminal": is_terminal
#         }
#         print('image:', obs['image'].shape)
#
#         # for key, value in obs.items():
#         #     space = self._obs_space[key]
#         #     if not isinstance(value, np.ndarray):
#         #         value = np.array(value)
#         #     # assert value in space, (key, value, value.dtype, value.shape, space)
#         return obs
#
#
#     def _step(self, action):
#
#         self.objects[0].ApplyForceToCenter([discrete_action_x[int(action)] * 300, discrete_action_y[int(action)] * 300], True)
#
#         images = []
#         for i in range(300):
#             self.world.Step(TIME_STEP, 10, 10)
#
#         pygame.display.flip()
#         self.clock.tick(TARGET_FPS)
#
#         # Draw the world
#         self.screen.fill((0, 0, 0, 0))
#         for body in self.world.bodies:
#             for fixture in body.fixtures:
#                 fixture.shape.draw(body, fixture)
#
#         # Stop all objects
#         for body in self.world.bodies:
#             body.linearVelocity = (0., 0.)
#             body.angularVelocity = 0.
#
#         img = self.render(mode="rgb_array")
#         images.append(Image.fromarray(img))
#
#         reward = 0
#         done = False
#         return img, reward, done, {}
#
#     def step(self, action):
#         if action['reset'] or self._done:
#             self._done = False
#             image, _, _, _ = self.reset()
#             return self._obs(image, 0.0, is_first=True)
#         img, reward, _, _ = self._step(action['action'])
#         print('img:', img.shape)
#         self._done = self._length and self._step >= self._length
#         obs = self._obs(img, reward, is_last=bool(self._done), is_terminal=bool(self._done))
#         self._step += 1
#         return obs
#
#
#
#
