# SPDX-License-Identifier: Apache-2.0

import time
from abc import ABC, abstractmethod
from typing import List

import vllm
from vllm.v1.core.kv_cache_utils import PrefixCachingMetrics
from vllm.v1.metrics.stats import IterationStats, SchedulerStats


class StatLoggerBase(ABC):

    @abstractmethod
    def log(self, scheduler_stats: SchedulerStats,
            iteration_stats: IterationStats):
        ...


class LoggingStatLogger(StatLoggerBase):

    def __init__(self):
        self._reset(time.monotonic())
        self.prefix_caching_metrics = PrefixCachingMetrics()

    def _reset(self, now):
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []

vllm.v1.metrics.loggers.LoggingStatLogger.__init__ = LoggingStatLogger.__init__
vllm.v1.metrics.loggers.LoggingStatLogger._reset = LoggingStatLogger._reset
