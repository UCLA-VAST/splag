#!/usr/bin/python3
import collections
import contextlib
import itertools
import logging
import math
import os.path
import re
import json
import sys
import types
from typing import Iterable, List, Tuple

import matplotlib
import numpy as np
from absl import app, flags
import scipy.stats

FLAGS = flags.FLAGS

flags.DEFINE_string('pdf_dir', '', 'directory to which PDF files are written')
flags.DEFINE_string('png_dir', '', 'directory to which PNG files are written')
flags.register_validator(
    'pdf_dir',
    lambda value: value or FLAGS.png_dir,
    message='no output directory is specified',
)
flags.register_validator(
    'png_dir',
    lambda value: value or FLAGS.pdf_dir,
    message='no output directory is specified',
)
flags.DEFINE_list('pq_size', '', 'numpy data dump for pq size history')
flags.DEFINE_list(
    'bucket_distribution',
    '',
    'numpy data dump for bucket distribution history',
)
flags.DEFINE_string('metadata', '', 'metadata of pq size history')

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # isort:skip pylint:disable=all

SAVEFIG_KWARGS = {
    'dpi': 300,
    'bbox_inches': 'tight',
    'metadata': {
        'CreationDate': None,
    },
}

XTICK_ROTATION = 90


def def_var(key, value):
  print(f'{key} = {value:.2g}', file=sys.stderr)
  key = key.incr('_', '')
  print(f'\\newcommand\\{key}{{{value:.2g}}}')


@contextlib.contextmanager
def figure(name: str):
  logging.info('drawing figure %s', name)
  yield
  if FLAGS.pdf_dir:
    plt.savefig(
        os.path.join(FLAGS.pdf_dir, f'{name}.pdf'),
        transparent=True,
        **SAVEFIG_KWARGS,
    )
  if FLAGS.png_dir:
    plt.savefig(
        os.path.join(FLAGS.png_dir, f'{name}.png'),
        transparent=False,
        **SAVEFIG_KWARGS,
    )
  plt.close()


def main(argv: List[str]):
  # matplotlib.font_manager._rebuild()
  plt.rcParams['font.family'] = 'Linux Libertine'
  plt.rcParams['font.size'] = 14  # 2x
  # plt.rcParams['hatch.linewidth'] = BAR_KWARGS['linewidth']
  plt.rcParams["figure.figsize"] = (6, 2.5)

  draw(argv)


# https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
_ANSI_ESCAPE = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')


class QueryMetric(types.SimpleNamespace):
  """Metrics of one SSSP query."""
  kernel_time: float
  cycle_count: int
  spill_count: int
  push_count: int
  cvc_idle_count: List[int]
  read_hit: List[float]
  write_hit: List[float]
  visited_vertex_count: int
  discarded_by_update_vertex_count: int
  discarded_by_filter_vertex_count: int
  teps: float
  work_efficiency: float

  @property
  def visited_edge_count(self) -> int:
    return sum([
        self.visited_vertex_count,
        self.discarded_by_filter_vertex_count,
        self.discarded_by_update_vertex_count,
    ])


class DataPoints(types.SimpleNamespace):
  mean: List[float]
  stdev: List[float]

  def __init__(self):
    super().__init__()
    self.mean = []
    self.stdev = []

  def __iadd__(self, stats: Iterable[float]) -> 'DataPoints':
    stat_tuple = [x for x in stats]
    self.mean.append(scipy.stats.hmean(stat_tuple))
    self.stdev.append(scipy.stats.gstd(stat_tuple))
    return self


def _sample(history: np.array, size: int = 1000) -> Tuple[np.array, np.array]:
  step = history.shape[0] // min(size, history.shape[0])
  x = np.arange(0, history.shape[0], step, dtype=np.int64)
  y = history[::step]
  x[-1] = history.shape[0] - 1
  y[-1] = history[-1]
  return x, y


def draw(argv: List[str]) -> None:
  cgpq_chunk_size = 1024
  cgpq_chunk_megabytes = cgpq_chunk_size * 8 / 1e6

  with figure('bucket-distribution'):
    with open(FLAGS.metadata) as fp:
      metadata = json.load(fp)
    for filename in FLAGS.bucket_distribution:
      history = np.load(filename)
      logging.info('loaded file "%s"', filename)
      max_of_each_bucket = np.max(history, 0)
      logging.info(
          'budget/actual: %f',
          np.max(max_of_each_bucket) * max_of_each_bucket.size /
          np.sum(max_of_each_bucket))
      plt.plot(*_sample(history), label=filename.split('.', 1)[0])
      plt.xlabel('Number of Traversed Edges')
      plt.ylabel('Bucket Size')

  with figure('pq-size'):
    for filename in FLAGS.pq_size:
      history = np.load(filename)
      logging.info('loaded file "%s"', filename)
      name = filename.split('.', 1)[0]
      vertex_count = metadata[name]['nv']
      edge_count = metadata[name]['ne']
      plt.plot(
          *_sample(history),
          label=f'{name} |V|={vertex_count} |E|={edge_count}',
      )
    plt.xlabel('Number of Traversed Edges')
    plt.ylabel('Number of Active Vertices')
    plt.legend(loc='upper right')

  datasets = []
  cvc_idle = DataPoints()
  spill_percentage = DataPoints()
  visited_vertices_percentage = DataPoints()
  discarded_by_filter_percentage = DataPoints()
  read_hit = DataPoints()
  write_hit = DataPoints()
  raw_teps = DataPoints()
  uniq_teps = DataPoints()
  work_efficiency = DataPoints()

  for filename in argv[1:]:
    datasets.append(os.path.basename(filename).split('.', 1)[0])
    with open(filename) as fp:
      metrics: List[QueryMetric] = []
      for line in fp:
        line = _ANSI_ESCAPE.sub('', line)
        if not line.startswith('I'):
          continue
        line = line.split('] ')[-1].strip()
        items = line.split()
        if 'kernel time:' in line:
          metrics.append(QueryMetric())
          metrics[-1].kernel_time = float(items[2])
          metrics[-1].cvc_idle_count = []
          metrics[-1].read_hit = []
          metrics[-1].write_hit = []
          metrics[-1].spill_count = 0
        elif 'TEPS:' in line:
          metrics[-1].teps = float(items[1])
        elif '#idle' in line:
          metrics[-1].cvc_idle_count.append(
              int(items[2]) / metrics[-1].cycle_count * 100)
        elif '#edges visited:' in line:
          metrics[-1].work_efficiency = int(items[2]) / int(
              items[-1].rstrip(')'))
        elif '#vertices visited:' in line:
          metrics[-1].visited_vertex_count = int(items[2])
        elif '#discarded by update:' in line:
          metrics[-1].discarded_by_update_vertex_count = int(items[3])
        elif '#discarded by filter:' in line:
          metrics[-1].discarded_by_filter_vertex_count = int(items[3])
        elif '#push:' in line:
          metrics[-1].push_count = int(items[1])
        elif 'cycle count:' in line:
          metrics[-1].cycle_count = int(items[2])
        elif 'read hit   :' in line:
          metrics[-1].read_hit.append(int(items[3]) / int(items[5]) * 100)
        elif 'write hit  :' in line:
          metrics[-1].write_hit.append(int(items[3]) / int(items[5]) * 100)
        elif 'spill count  :' in line:
          metrics[-1].spill_count += int(items[3])

      cvc_idle += itertools.chain.from_iterable(
          m.cvc_idle_count for m in metrics)
      spill_percentage += (m.spill_count * cgpq_chunk_size * 100 / m.push_count
                           for m in metrics
                           if m.spill_count > 0)
      visited_vertices_percentage += (
          m.visited_vertex_count * 100 / m.visited_edge_count for m in metrics)
      discarded_by_filter_percentage += (m.discarded_by_filter_vertex_count *
                                         100 / m.visited_edge_count
                                         for m in metrics)
      read_hit += itertools.chain.from_iterable(m.read_hit for m in metrics)
      write_hit += itertools.chain.from_iterable(m.write_hit for m in metrics)
      raw_teps += (m.visited_edge_count / m.kernel_time / 1e6 for m in metrics)
      uniq_teps += (m.teps / 1e6 for m in metrics)
      work_efficiency += (m.work_efficiency for m in metrics)

  with figure('cvc-idle'):
    plt.bar(
        datasets,
        cvc_idle.mean,
    )
    plt.xlabel('Dataset')
    plt.xticks(rotation=XTICK_ROTATION)
    plt.ylabel('CVC Idling (%)')

  with figure('spill-stack'):
    plt.bar(
        datasets,
        spill_percentage.mean,
    )
    plt.xlabel('Dataset')
    plt.xticks(rotation=XTICK_ROTATION)
    plt.ylabel('Spilled Vertices (%)')

  with figure('cache-rate'):
    plt.errorbar(
        datasets,
        read_hit.mean,
        fmt='o',
        label='Read',
    )
    plt.errorbar(
        datasets,
        write_hit.mean,
        fmt='s',
        label='Write',
    )
    plt.xlabel('Dataset')
    plt.xticks(rotation=XTICK_ROTATION)
    plt.ylabel('Cache Hit Rate (%)')
    plt.legend()

  with figure('discarded-vertices'):
    plt.bar(
        datasets,
        visited_vertices_percentage.mean,
        bottom=discarded_by_filter_percentage.mean,
        label='Processed by edge fetcher',
    )
    plt.bar(
        datasets,
        discarded_by_filter_percentage.mean,
        label='Discarded by CVC filtering',
    )

    plt.xlabel('Dataset')
    plt.xticks(rotation=XTICK_ROTATION)
    plt.ylabel('Active Vertices (%)')
    plt.legend()

  with figure('teps'):
    plt.errorbar(
        datasets,
        raw_teps.mean,
        fmt='o',
        label='Traversal',
    )
    plt.errorbar(
        datasets,
        uniq_teps.mean,
        fmt='s',
        label='Algorithm',
    )
    plt.xlabel('Dataset')
    plt.xticks(rotation=XTICK_ROTATION)
    plt.ylabel('Throughput (MTEPS)')
    plt.legend()

  with figure('work-efficiency'):
    plt.bar(
        datasets,
        work_efficiency.mean,
    )
    plt.xlabel('Dataset')
    plt.xticks(rotation=XTICK_ROTATION)
    plt.ylabel('Amount of Work')


if __name__ == '__main__':
  app.run(main)
