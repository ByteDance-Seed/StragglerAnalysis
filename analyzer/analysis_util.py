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

import gc
from typing import Callable, List

import pandas as pd

from analyzer.metatypes import OpType


def get_step_time(df: pd.DataFrame):
    df = df.groupby('step').agg({'start_ts': 'min', 'end_ts': 'max'})
    return (df['end_ts'] - df['start_ts']).rename('step_time').reset_index()


def get_step_time_ratios(df_ori: pd.DataFrame = None,
                         df_new: pd.DataFrame = None,
                         step_time_original=None,
                         step_time_new=None):
    """df_ori.step_time / df_new.step_time, and mean_ori / mean_new, i.e., per-step time ratio and average step time ratio"""
    if step_time_original is None:
        step_time_original = get_step_time(df_ori)
    if step_time_new is None:
        step_time_new = get_step_time(df_new)
    step_time_original = step_time_original.set_index('step')
    step_time_new = step_time_new.set_index('step')
    step_ratio = (step_time_original.step_time / step_time_new.step_time).sort_values(ascending=False).to_dict()
    return step_ratio, step_time_original.step_time.mean() / step_time_new.step_time.mean()


def simulation_analysis_reduce_dim_v4(sim: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame,
                                      reduce_dim: List[str]):
    all_dims = set(['dp_rank', 'stage', 'step'])
    if reduce_dim is not None:
        non_reduce_dim = list(all_dims - set(reduce_dim))
        df_no_variation = df.copy()
        comm_optypes = OpType.p2p_optypes() + OpType.dp_comm_optypes() + [
            OpType.layernorm_grads_all_reduce, OpType.embedding_grads_all_reduce, OpType.optimizer_clip_main_grad
        ]
        is_comm = df.optype.isin(comm_optypes)
        # note that this is different from df.is_comm since is_comm here is conceptual
        # while df.is_comm is for simulation implementation to indicate if an op requires sync with others
        is_compute = ~is_comm
        # compute
        df_mean = df[is_compute].groupby(['optype'] + non_reduce_dim,
                                         observed=False)[['duration', 'transfer_duration']].transform('mean')
        df_no_variation.loc[is_compute,
                            ['duration', 'transfer_duration']] = df_mean.loc[is_compute,
                                                                             ['duration', 'transfer_duration']]
        # commm
        df_median = df[is_comm].groupby(['optype'] + non_reduce_dim,
                                        observed=False)[['duration', 'transfer_duration']].transform('median')
        df_no_variation.loc[is_comm,
                            ['duration', 'transfer_duration']] = df_median.loc[is_comm,
                                                                               ['duration', 'transfer_duration']]
    else:
        df_no_variation = df.copy()
    df_no_variation, step_time_no_variation = sim(df_no_variation)  # could be inplace
    return (*get_step_time_ratios(df, df_no_variation, None, step_time_no_variation), df_no_variation,
            step_time_no_variation)


def simulation_analysis_reduce_dim(
        sim: Callable[[pd.DataFrame], pd.DataFrame],
        df: pd.DataFrame,
        reduce_dim: List[str],
        agg_method='mean',  # TODO(JK): remove this field.
        version='latest'):

    if version == 'v4' or version == 'latest':
        func = simulation_analysis_reduce_dim_v4
    else:
        raise ValueError(f"Unrecognized version: {version}")
    return func(sim, df, reduce_dim)


def simulation_analysis_no_delay(sim: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame):
    df_no_delay = df.copy()
    df_no_delay['launch_delay'] = 0.
    df_no_delay, step_time_no_delay = sim(df_no_delay)  # could be inplace
    return (*get_step_time_ratios(df, df_no_delay, None, step_time_no_delay), df_no_delay, step_time_no_delay)


def set_op_id_(df: pd.DataFrame):
    ops = {op: i for i, op in enumerate(df.op.unique())}
    df['opid'] = df['op'].map(ops)


def compress_dtype(df: pd.DataFrame, compress_start_end_ts=False, cols=None):
    dtypes = {
        'dp_rank': ['int8', 'int16', 'int32'],
        'stage': ['int8', 'int16', 'int32'],
        'rank': ['int16', 'int32'],
        'seq_id': 'int32',
        'mc': ['int8', 'int16', 'int32'],
        'mb_id': ['int8', 'int16', 'int32'],
        'gmc': ['int16', 'int32'],
        'opid': ['int16', 'int32'],
        'launch_delay': 'float32',
        'transfer_duration': 'float32',
        'duration': 'float32',
    }
    for key, dtype in dtypes.items():
        if key not in df.columns or (cols is not None and key not in cols):
            continue
        if isinstance(dtype, str) and dtype.startswith('float'):
            df[key] = df[key].astype(dtype)
            continue
        if not isinstance(dtype, list):
            dtype = [dtype]
        for dt in dtype:
            assert dt.startswith('int')
            assert not pd.isnull(df[key]).any(), f"NA found in '{key}' column:\n{df[pd.isnull(df[key])]}"
            if (df[key].astype(dt) == df[key]).all():
                df[key] = df[key].astype(dt)
                break

    # offset such that start_ts of each step is 0, to enable lower-precision float32 for start_ts and end_ts
    if compress_start_end_ts:
        df.attrs['step_start'] = df.groupby('step')['start_ts'].min().to_dict()
        min_start_ts = df.groupby('step')['start_ts'].transform('min')
        df.start_ts -= min_start_ts
        df.end_ts -= min_start_ts
        df.start_ts = df.start_ts.astype('float32')
        df.end_ts = df.end_ts.astype('float32')
    gc.collect()
    return df
