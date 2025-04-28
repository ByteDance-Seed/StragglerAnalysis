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

import logging
import sys
import os
from typing import Union


class AnalysisLogger:

    def __new__(cls):
        if not hasattr(cls, "instance"):
            level: Union[int, str] = logging.getLevelName(os.getenv("ANALYSIS_LOG_LEVEL", "INFO"))
            if isinstance(level, str):
                level = logging.INFO
            formatter = logging.Formatter(
                "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler = logging.StreamHandler(stream=sys.stderr)
            handler.setFormatter(formatter)
            cls.instance = logging.getLogger("analysis")
            cls.instance.handlers.clear()
            cls.instance.addHandler(handler)
            cls.instance.setLevel(level)
            cls.instance.propagate = False
        return cls.instance
