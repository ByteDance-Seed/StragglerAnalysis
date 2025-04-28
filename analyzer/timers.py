# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import perf_counter

from analysis_logger import AnalysisLogger


class RecordTime:
    timers = {}

    def __init__(self, name=None, print_res=False) -> None:
        self.print_res = print_res
        self.name = name

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        RecordTime.timers[self.name] = self.time
        if self.print_res:
            prefix = f'Time of `{self.name}`: ' if self.name is not None else 'Time: '
            AnalysisLogger().info(f'{prefix}{self.time:.3f} seconds')
