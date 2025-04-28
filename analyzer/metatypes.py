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

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class JobMeta:
    trial_name: str
    run_id: int = 0
    dp_size: int = 1
    pp_size: int = 1
    tp_size: int = 1
    dsp_size: int = 1
    vpp_size: int = 1
    world_size: int = 1

    def __post_init__(self):
        for field_name in self.__dict__:
            field_content = self.__dict__[field_name]
            if field_name.endswith("size") and (not isinstance(field_content, int) or field_content <= 0):
                raise ValueError(
                    f"JobMeta instance's {field_name}={field_content}, {type(field_content)}, expected positive integer"
                )
        assert isinstance(self.run_id, int) and self.run_id >= 0
        assert self.world_size == self.dp_size * self.pp_size * self.tp_size * self.dsp_size
        assert len(self.trial_name) > 0


@dataclass
class AnalyzerResult:
    """
    AnalyzerResult

    This class represents the results of an analysis.
    
    Fields starting with `launch_` are for simulation error analysis and mostly for internal use
    (named `launch` as it's caused by launch-delay. Refer to `calc_launch_delay` for detail).
    
    Fields starting with `blocking_` are for straggler analysis (named `blocking` as stragglers blocks 
    other workers at synchronization).
    """

    job_meta: JobMeta

    ########## step -> time ##########
    # original step time
    step_t_base: Dict[int, float]

    # step time simulated by only removing all launch delays
    step_t_nodelay: Dict[int, float]

    # step time simulated by both removing all launch delays and replacing duration with means/medians
    step_t_noblk: Dict[int, float] 


    ########## slowdowns ##########
    # effect of launch-delay, measured by step_t_base.mean() / step_t_nodelay.mean()
    launch_slowdown: float

    # straggler-slowdown (Eq. (1) of paper), measured by step_t_nodelay.mean() / step_t_noblk.mean()
    blocking_slowdown: float

    # per-step slowdown
    blocking_step_slowdown: Dict[int, float]
    launch_step_slowdown: Dict[int, float]


    ########## Attribution to DP rank, PP rank and OpType ##########
    # blocking_deompose_by_X[i] := step_t_nodelay_but_Xi.mean() / step_t_noblk.mean(),
    # step_t_nodelay_but_Xi: similar to `step_t_nodelay` except keeping ops in Xi with original duration
    # For launch_decompose_by_X, it's the same, but keeping the launch-delay (which will be simulated) instead of duration.
    # It's more costly to run and if blocking/launch_slowdown is too small we currently skip this analysis and set to None.

    # dp_rank -> slowdown
    blocking_decompose_by_dp_rank: Optional[Dict[int, float]] = None
    launch_decompose_by_dp_rank: Optional[Dict[int, float]] = None
    # stage -> slowdown
    blocking_decompose_by_stage: Optional[Dict[int, float]] = None
    launch_decompose_by_stage: Optional[Dict[int, float]] = None
    # optype -> slowdown
    blocking_decompose_by_optype: Optional[Dict[str, float]] = None
    launch_decompose_by_optype: Optional[Dict[str, float]] = None

    # per-step attribution result
    # step -> dp_rank -> slowdown
    blocking_decompose_by_dp_rank_per_step: \
        Optional[Dict[int, Dict[int, float]]] = None
    launch_decompose_by_dp_rank_per_step: \
        Optional[Dict[int, Dict[int, float]]] = None
    # step -> stage -> slowdown
    blocking_decompose_by_stage_per_step: \
        Optional[Dict[int, Dict[int, float]]] = None
    launch_decompose_by_stage_per_step: \
        Optional[Dict[int, Dict[int, float]]] = None
    # step -> optype -> slowdown
    blocking_decompose_by_optype_per_step: \
        Optional[Dict[int, Dict[str, float]]] = None
    launch_decompose_by_optype_per_step: \
        Optional[Dict[int, Dict[str, float]]] = None


class OpType:
    forward_compute = 'forward-compute'
    backward_compute = 'backward-compute'
    grads_reduce_scatter = 'grads-reduce-scatter'
    layernorm_grads_all_reduce = 'layernorm-grads-all-reduce'
    embedding_grads_all_reduce = 'embedding-grads-all-reduce'
    separate_grads_all_reduce = 'separate-grads-all-reduce'
    optimizer_clip_main_grad = 'optimizer-clip-main-grad'
    optimizer = 'optimizer'
    params_all_gather = 'params-all-gather'
    gc = 'gc'  # TODO(JK): remove this field and from dependency model
    forward_send = 'forward-send'
    forward_recv = 'forward-recv'
    backward_send = 'backward-send'
    backward_recv = 'backward-recv'

    @staticmethod
    def p2p_optypes() -> List[str]:
        return [
            OpType.forward_send,
            OpType.forward_recv,
            OpType.backward_send,
            OpType.backward_recv,
        ]

    @staticmethod
    def dp_comm_optypes() -> List[str]:
        return [
            OpType.params_all_gather,
            OpType.separate_grads_all_reduce,
            OpType.grads_reduce_scatter,
        ]
