import logging
from typing import Optional
from typing import Union
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import os

DIR_X = 0
DIR_Y = 1


class Segment:
  def __init__(self, direction, start, stop, value, start_audio_time, stop_audio_time):
    # both start and stop is in minutes
    self.direction = direction
    # start/stop axis time
    self.start = start
    self.stop = stop
    self.value = value
    self.start_audio_time = start_audio_time
    self.stop_audio_time = stop_audio_time

  def __str__(self):
    if self.direction == DIR_X:
      p1 = (self.start, self.value)
      p2 = (self.stop, self.value)
    elif self.direction == DIR_Y:
      p1 = (self.value, self.start)
      p2 = (self.value, self.stop)
    else:
      assert False

    return f'({p1}, {p2})'


def fit_to_get_k(segments) -> Optional[Union[str, float]]:
  points = []
  for segment in segments:
    b = segment.start
    while True:
      if b > segment.stop:
        break
      else:
        if segment.direction == DIR_X:
          point = (b, segment.value)
        elif segment.direction == DIR_Y:
          point = (segment.value, b)
        else:
          assert False
        points.append(point)
        b += 1 / 60
  if len(points) <= 1:
    return None
  else:
    xs = []
    ys = []
    for x, y in points:
      xs.append([x])
      ys.append(y)
    x_std = np.std(xs)
    if x_std <= 1e-6:
      return 'inf'
    else:
      model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
      model.fit(xs, ys)
      return model.coef_[0]


class Display:

  def __init__(self):
    # counts of consecutive lines
    self.teacher_count = 0
    self.student_count = 0
    self.num_angles = 0
    # the unit for segment is minutes, not seconds
    self.segments = []

    # tip of the curve, in seconds
    self._last_x = 0.0
    self._last_y = 0.0
    self._last_direction = None
    self._last_stop_audio_time = 0.0

  def axis_limit(self):
    return max(self._last_x, self._last_y) / 60.0

  def _update_axis(self, direction, span, start_audio_time, stop_audio_time):
    m = 60.0
    if direction == DIR_X:
      b = self._last_x
      v = self._last_y
      e = b + span
      self._last_x = e
      self.teacher_count += 1
    elif direction == DIR_Y:
      b = self._last_y
      v = self._last_x
      e = b + span
      self._last_y = e
      self.student_count += 1
    else:
      # programming error
      assert False

    # count number of angles
    if self._last_direction is None:
      self._last_direction = direction
    if self._last_direction != direction:
      self.num_angles += 1
      self._last_direction = direction

    segment = Segment(direction, b / m, e / m, v / m, start_audio_time, stop_audio_time)

    return segment

  def feed_concrete(self, start_audio_time, stop_audio_time, speaker_name) -> List:
    """
    Feed a "concrete" data point
    (v.s. inferred data points, i.e. the gap between two concrete points)
    """
    logger = logging.getLogger(__name__)

    if start_audio_time > stop_audio_time:
      return []

    segments = []

    # account for the gap
    start_audio_gap = self._last_stop_audio_time
    stop_audio_gap = start_audio_time
    gap_span = max(0.0, stop_audio_gap - start_audio_gap)
    self._last_stop_audio_time = stop_audio_time

    if gap_span <= 0:
      pass
    else:
      # handling gaps
      # (max 97-percentile of st, ss, tt, ts is about 3.8 seconds)
      if gap_span >= 5.0:
        # the gap is long enough, treat it as student
        segment = self._update_axis(DIR_Y, gap_span, start_audio_gap, stop_audio_gap)
        segments.append(segment)
      else:
        # gap < 5.0
        if (
            speaker_name == 't'
            and self._last_direction == DIR_X
        ):
          # there is a small gap between two teachers,
          # treat the gap as teacher
          segment = self._update_axis(DIR_X, gap_span, start_audio_gap, stop_audio_gap)
          segments.append(segment)
        else:
          # the gap is small, but is not between two teachers
          # treat it as student
          segment = self._update_axis(DIR_Y, gap_span, start_audio_gap, stop_audio_gap)
          segments.append(segment)

    # process this segment
    span = max(0.0, stop_audio_time - start_audio_time)
    if span > 0:
      if speaker_name == 't':
        segment = self._update_axis(DIR_X, span, start_audio_time, stop_audio_time)
        segments.append(segment)
      elif speaker_name == 's':
        segment = self._update_axis(DIR_Y, span, start_audio_time, stop_audio_time)
        segments.append(segment)
      elif speaker_name == dt.Speaker.UNKNOWN.name:
        # unknown speaker is dropped
        assert False
      else:
        # programming error
        assert False

    self.segments.extend(segments)

    return segments

  def finish(self, audio_length_in_seconds):
    # make up a silence data point
    start = self._last_stop_audio_time
    stop = audio_length_in_seconds
    speaker_name = 's'
    return self.feed_concrete(start, stop, speaker_name)



class NaiveDisplay:

  def __init__(self):
    # counts of consecutive lines
    self.teacher_count = 0
    self.student_count = 0
    self.num_angles = 0
    # the unit for segment is minutes, not seconds
    self.segments = []

    # tip of the curve, in seconds
    self._last_x = 0.0
    self._last_y = 0.0
    self._last_direction = None
    self._last_stop_audio_time = 0.0

  def axis_limit(self):
    return max(self._last_x, self._last_y) / 60.0

  def _update_axis(self, direction, span, start_audio_time, stop_audio_time):
    m = 60.0
    if direction == DIR_X:
      b = self._last_x
      v = self._last_y
      e = b + span
      self._last_x = e
      self.teacher_count += 1
    elif direction == DIR_Y:
      b = self._last_y
      v = self._last_x
      e = b + span
      self._last_y = e
      self.student_count += 1
    else:
      # programming error
      assert False

    # count number of angles
    if self._last_direction is None:
      self._last_direction = direction
    if self._last_direction != direction:
      self.num_angles += 1
      self._last_direction = direction

    segment = Segment(direction, b / m, e / m, v / m, start_audio_time, stop_audio_time)

    return segment

  def feed_concrete(self, start_audio_time, stop_audio_time, speaker_name) -> List:

    if start_audio_time > stop_audio_time:
      return []

    segments = []

    # process this segment
    span = max(0.0, stop_audio_time - start_audio_time)
    if span > 0:
      if speaker_name == 't':
        segment = self._update_axis(DIR_X, span, start_audio_time, stop_audio_time)
        segments.append(segment)
      elif speaker_name == 's':
        segment = self._update_axis(DIR_Y, span, start_audio_time, stop_audio_time)
        segments.append(segment)
      else:
        # programming error
        assert False

    self.segments.extend(segments)

    return segments

  def finish(self, _):
    return []


def plot(rowss, names, save_path, hints):
  name = os.path.basename(save_path)
  fig = plt.figure()

  names = [
    os.path.basename(n)
    for n in names
  ]
  names = [
    os.path.splitext(n)[0]
    for n in names
  ]
  title = '\nvs. '.join(names)
  if hints:
    hint = '\n'.join(hints)
    title = hint + '\n' + title
  fig.suptitle(title, x=0, ha='left')

  fig.gca().set_aspect('equal')

  colors = ['black', 'blue', 'yellow']
  assert len(rowss) <= len(colors)

  lims = []
  for rows, color in zip(rowss, colors):
    rows.sort(key=lambda t: t[0])
    audio_length_in_seconds = np.max([stop for _, stop, _ in rows])

    display = NaiveDisplay()
    for start_audio_time, stop_audio_time, speaker_name in rows:
      display.feed_concrete(start_audio_time, stop_audio_time, speaker_name)
    display.finish(audio_length_in_seconds)
    lims.append(display.axis_limit())

    for segment in display.segments:
      if segment.direction == DIR_X:
        plt.plot([segment.start, segment.stop], [segment.value, segment.value], '-', color=color)
      elif segment.direction == DIR_Y:
        plt.plot([segment.value, segment.value], [segment.start, segment.stop], '-', color=color)
      else:
        assert False

  plt.xlim([0, np.max(lims)])
  plt.ylim([0, np.max(lims)])
  plt.grid(True)
  plt.tight_layout()

  plt.savefig(save_path)
  plt.close(fig)
