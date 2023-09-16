import warnings
import dreamerv3
from dreamerv3 import embodied
import gym

def main():
  import elastic2d
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': '~/logdir/mini_circle_cont_slow',
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      # 'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      # embodied.logger.TensorBoardOutput(logdir),
      embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  NUMOFOBJS = 5
  EVAL_NUMOFOBJS = 5

  OBJS = [
      "circle" #, "triangle", "square"
  ]

  # Object color (EXCEPT WHITE)
  COLORS = [
      [0, 255, 0, 255],  # LIME
      [0, 0, 255, 255],  # BLUE
      # [0, 255, 255, 255], # CYAN
      # [0, 127, 127, 255], # TEAL
      # [0, 0, 127, 255],   # NAVY
      # [0, 127, 0, 255]    # GREEN
  ]

  SIZES = [2, 2]
  cleanup = []
  # import crafter
  from embodied.envs import from_gym
  env = gym.make('elastic2d-v1', colors=COLORS, objs=OBJS, numofobjs=NUMOFOBJS, sizes=SIZES)  # Replace this with your Gym env.
  env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  eval_env = gym.make('elastic2d-v1', colors=COLORS, objs=OBJS, numofobjs=EVAL_NUMOFOBJS, sizes=SIZES)
  eval_env = from_gym.FromGym(eval_env, obs_key='image')  # Or obs_key='vector'.
  eval_env = dreamerv3.wrap_env(eval_env, config)
  eval_env = embodied.BatchEnv([eval_env], parallel=False)


  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
       config.batch_length, config.replay_size, logdir / 'replay')
  eval_replay = embodied.replay.Uniform(
       config.batch_length, config.replay_size, logdir / 'eval_replay_{}_'.format(EVAL_NUMOFOBJS))
  # replay = make_replay(config, logdir / 'replay')
  # eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  # embodied.run.train_eval(agent, env, eval_env, replay, eval_replay, logger, args)
  embodied.run.eval_only(agent, eval_env, eval_replay, logger, args)
  cleanup += [env, eval_env]
  for obj in cleanup:
      obj.close()

def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, **kwargs):
  assert config.replay == 'uniform' or not rate_limit
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform' or is_eval:
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay

if __name__ == '__main__':
  main()
