import collections
import concurrent.futures
import datetime
import json
import os
import re
import time

import numpy as np

from . import path


class Logger:

  def __init__(self, step, outputs, multiplier=1):
    assert outputs, 'Provide a list of logger outputs.'
    self.step = step
    self.outputs = outputs
    self.multiplier = multiplier
    self._last_step = None
    self._last_time = None
    self._metrics = []

  def add(self, mapping, prefix=None):
    step = int(self.step) * self.multiplier
    for name, value in dict(mapping).items():
      name = f'{prefix}/{name}' if prefix else name
      value = np.asarray(value)
      if len(value.shape) not in (0, 1, 2, 3, 4):
        raise ValueError(
            f"Shape {value.shape} for name '{name}' cannot be "
            "interpreted as scalar, histogram, image, or video.")
      self._metrics.append((step, name, value))

  def scalar(self, name, value):
    self.add({name: value})

  def image(self, name, value):
    self.add({name: value})

  def video(self, name, value):
    self.add({name: value})

  def write(self, fps=False):
    if fps:
      value = self._compute_fps()
      if value is not None:
        self.scalar('fps', value)
    if not self._metrics:
      return
    for output in self.outputs:
      output(tuple(self._metrics))
    self._metrics.clear()

  def _compute_fps(self):
    step = int(self.step) * self.multiplier
    if self._last_step is None:
      self._last_time = time.time()
      self._last_step = step
      return None
    steps = step - self._last_step
    duration = time.time() - self._last_time
    self._last_time += duration
    self._last_step = step
    return steps / duration


class AsyncOutput:

  def __init__(self, callback, parallel=True):
    self._callback = callback
    self._parallel = parallel
    if parallel:
      self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
      self._future = None

  def __call__(self, summaries):
    if self._parallel:
      self._future and self._future.result()
      self._future = self._executor.submit(self._callback, summaries)
    else:
      self._callback(summaries)


class TerminalOutput:

  def __init__(self, pattern=r'.*', name=None):
    self._pattern = re.compile(pattern)
    self._name = name
    try:
      import rich.console
      self._console = rich.console.Console()
    except ImportError:
      self._console = None

  def __call__(self, summaries):
    step = max(s for s, _, _, in summaries)
    scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
    scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
    formatted = {k: self._format_value(v) for k, v in scalars.items()}
    if self._console:
      if self._name:
        self._console.rule(f'[green bold]{self._name} (Step {step})')
      else:
        self._console.rule(f'[green bold]Step {step}')
      self._console.print(' [blue]/[/blue] '.join(
          f'{k} {v}' for k, v in formatted.items()))
      print('')
    else:
      message = ' / '.join(f'{k} {v}' for k, v in formatted.items())
      message = f'[{step}] {message}'
      if self._name:
        message = f'[{self._name}] {message}'
      print(message, flush=True)

  def _format_value(self, value):
    value = float(value)
    if value == 0:
      return '0'
    elif 0.01 < abs(value) < 10000:
      value = f'{value:.2f}'
      value = value.rstrip('0')
      value = value.rstrip('0')
      value = value.rstrip('.')
      return value
    else:
      value = f'{value:.1e}'
      value = value.replace('.0e', 'e')
      value = value.replace('+0', '')
      value = value.replace('+', '')
      value = value.replace('-0', '-')
    return value


class JSONLOutput(AsyncOutput):

  def __init__(
      self, logdir, filename='metrics.jsonl', pattern=r'.*', parallel=True):
    super().__init__(self._write, parallel)
    self._filename = filename
    self._pattern = re.compile(pattern)
    self._logdir = path.Path(logdir)
    self._logdir.mkdirs()

  def _write(self, summaries):
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      if len(value.shape) == 0 and self._pattern.search(name):
        bystep[step][name] = float(value)
    lines = ''.join([
        json.dumps({'step': step, **scalars}) + '\n'
        for step, scalars in bystep.items()])
    with (self._logdir / self._filename).open('a') as f:
      f.write(lines)


class TensorBoardOutput(AsyncOutput):

  def __init__(self, logdir, fps=20, maxsize=1e9, parallel=True):
    super().__init__(self._write, parallel)
    self._logdir = str(logdir)
    if self._logdir.startswith('/gcs/'):
      self._logdir = self._logdir.replace('/gcs/', 'gs://')
    self._fps = fps
    self._writer = None
    self._maxsize = self._logdir.startswith('gs://') and maxsize
    if self._maxsize:
      self._checker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
      self._promise = None

  def _write(self, summaries):
    import tensorflow as tf
    reset = False
    if self._maxsize:
      result = self._promise and self._promise.result()
      # print('Current TensorBoard event file size:', result)
      reset = (self._promise and result >= self._maxsize)
      self._promise = self._checker.submit(self._check)
    if not self._writer or reset:
      print('Creating new TensorBoard event file writer.')
      self._writer = tf.summary.create_file_writer(
          self._logdir, flush_millis=1000, max_queue=10000)
    self._writer.set_as_default()
    for step, name, value in summaries:
      try:
        if len(value.shape) == 0:
          tf.summary.scalar(name, value, step)
        elif len(value.shape) == 1:
          if len(value) > 1024:
            value = value.copy()
            np.random.shuffle(value)
            value = value[:1024]
          tf.summary.histogram(name, value, step)
        elif len(value.shape) == 2:
          tf.summary.image(name, value, step)
        elif len(value.shape) == 3:
          tf.summary.image(name, value, step)
        elif len(value.shape) == 4:
          self._video_summary(name, value, step)
      except Exception:
        print('Error writing summary:', name)
        raise
    self._writer.flush()

  def _check(self):
    import tensorflow as tf
    events = tf.io.gfile.glob(self._logdir.rstrip('/') + '/events.out.*')
    return tf.io.gfile.stat(sorted(events)[-1]).length if events else 0

  def _video_summary(self, name, video, step):
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
      video = np.clip(255 * video, 0, 255).astype(np.uint8)
    try:
      T, H, W, C = video.shape
      summary = tf1.Summary()
      image = tf1.Summary.Image(height=H, width=W, colorspace=C)
      image.encoded_image_string = _encode_gif(video, self._fps)
      summary.value.add(tag=name, image=image)
      tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
      print('GIF summaries require ffmpeg in $PATH.', e)
      tf.summary.image(name, video, step)


class WandBOutput:

  def __init__(self, name, config, pattern=r'.*'):
    self._pattern = re.compile(pattern)
    import wandb
    from collections import deque

    wandb.init(
        project="dreamerv3",
        name=name,
        # sync_tensorboard=True,,
        # entity='word-bots',
        config=dict(config),
    )
    self._wandb = wandb
    self.epi_eval_buffer = deque(maxlen=100)
    self.epi_report_buffer = deque(maxlen=100)

  def __call__(self, summaries):
    import matplotlib.pyplot as plt
    import cv2
    bystep = collections.defaultdict(dict)
    wandb = self._wandb
    accuracy = []
    for step, name, value in summaries:
      # print('name:', name)
      if "openl_accuracy" in name:
        accuracy = value
      elif 'openl_epi_accuracy' in name:
        # data = [[i+1, v] for (i, v) in enumerate(value)]
        # table = wandb.Table(data=data, columns=["epi", "accuracy"])
        # fields = {"x": "epi", "value": "accuracy"}
        # custom_graph = wandb.plot.line(table, x='epi', y='accuracy', title='epi_accuracy')
        # self._wandb.log({'epi_accuracy_line' : custom_graph})
        # buf = io.BytesIO()
        if 'eval' in name:
          flag = 'eval'
          self.epi_eval_buffer.append(value)
          buffer = np.array(self.epi_eval_buffer)
        else:
          flag = 'report'
          self.epi_report_buffer.append(value)
          buffer = np.array(self.epi_report_buffer)
        # print('buffer_shape:', buffer.shape)
        avg_buffer = np.average(buffer, axis=0) # 5, 9, 14, 19 data is needed.
        avg_value = np.average(buffer)
        fig, ax = plt.subplots()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["lines.markersize"] = 8
        plt.subplots_adjust(left=.14, right=.88, bottom=0.14, top=0.98)
        ax.grid(which='major', axis='y', linestyle='--')

        ax.set_xlabel("Steps", fontsize=18)
        ax.set_ylabel("Accuracy", fontsize=18)
        line_color1 = '#5691cc'
        line_color2 = '#db4d50'
        lns1 = ax.plot(np.arange(1, 22), np.array(value), color=line_color1, linewidth=2, label='Current')
        lns2 = ax.plot(np.arange(1, 22), np.array(avg_buffer), color=line_color2, linewidth=2, label='100steps avg')

        ax.set_xticks(np.arange(1, 25, 4))
        ax.set_xticklabels(["1", "5", "9", "13", "17", "21"], fontsize=16)
        ax.set_yticks(np.arange(0, 120, 20))
        ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontsize=16)
        ax.set_ylim(0, 100)
        ax.set_xlim(1, 22)
        ax.text(5, round(avg_buffer[5], 2), round(avg_buffer[5], 2), size=12)
        for index in range(9, 24, 5):
          ax.text(index, round(avg_buffer[index], 2), round(avg_buffer[index], 2), size=12)

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')
        # print('value_shape:', np.array(value).shape)
        # plt.savefig(buf, format="png")
        plt.savefig("/home/workspace/dreamerv3-main/graph/{}.png".format(flag))
        plt.close()
        img = cv2.imread("/home/workspace/dreamerv3-main/graph/{}.png".format(flag), cv2.IMREAD_COLOR)
        img = np.array(img)
        print('graph_shape:', img.shape)
        self._wandb.log({flag: wandb.Image(img)}, step=step)


      if len(value.shape) == 0 and self._pattern.search(name):
        bystep[step][name] = float(value)
      elif len(value.shape) == 1:
        bystep[step][name] = wandb.Histogram(value)
      elif len(value.shape) == 2:
        value = np.clip(255 * value, 0, 255).astype(np.uint8)
        value = np.transpose(value, [2, 0, 1])
        bystep[step][name] = wandb.Image(value)
      elif len(value.shape) == 3:
        value = np.clip(255 * value, 0, 255).astype(np.uint8)
        value = np.transpose(value, [2, 0, 1])
        bystep[step][name] = wandb.Image(value)
      elif len(value.shape) == 4:
        # Sanity check that the channeld dimension is last
        assert value.shape[3] in [1, 3, 4], f"Invalid shape: {value.shape}"
        value = np.transpose(value, [0, 3, 1, 2])
        # If the video is a float, convert it to uint8
        if np.issubdtype(value.dtype, np.floating):
          value = np.clip(255 * value, 0, 255).astype(np.uint8)
        bystep[step][name+'one_step'] = wandb.Image(np.transpose(value[5], [1, 2, 0]))
        bystep[step][name] = wandb.Video(value, fps=3, caption=np.array2string(np.array(accuracy)))

    for step, metrics in bystep.items():
      self._wandb.log(metrics, step=step)


class MLFlowOutput:

  def __init__(self, run_name=None, resume_id=None, config=None, prefix=None):
    import mlflow
    self._mlflow = mlflow
    self._prefix = prefix
    self._setup(run_name, resume_id, config)

  def __call__(self, summaries):
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      if len(value.shape) == 0 and self._pattern.search(name):
        name = f'{self._prefix}/{name}' if self._prefix else name
        bystep[step][name] = float(value)
    for step, metrics in bystep.items():
      self._mlflow.log_metrics(metrics, step=step)

  def _setup(self, run_name, resume_id, config):
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'local')
    run_name = run_name or os.environ.get('MLFLOW_RUN_NAME')
    resume_id = resume_id or os.environ.get('MLFLOW_RESUME_ID')
    print('MLFlow Tracking URI:', tracking_uri)
    print('MLFlow Run Name:    ', run_name)
    print('MLFlow Resume ID:   ', resume_id)
    if resume_id:
      runs = self._mlflow.search_runs(None, f'tags.resume_id="{resume_id}"')
      assert len(runs), ('No runs to resume found.', resume_id)
      self._mlflow.start_run(run_name=run_name, run_id=runs['run_id'].iloc[0])
      for key, value in config.items():
        self._mlflow.log_param(key, value)
    else:
      tags = {'resume_id': resume_id or ''}
      self._mlflow.start_run(run_name=run_name, tags=tags)


def _encode_gif(frames, fps):
  from subprocess import Popen, PIPE
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = ' '.join([
      'ffmpeg -y -f rawvideo -vcodec rawvideo',
      f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
      '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      f'-r {fps:.02f} -f gif -'])
  proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in frames:
    proc.stdin.write(image.tobytes())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
  del proc
  return out
