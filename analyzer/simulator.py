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

from collections import deque
from typing import Dict, List

import numpy as np
import pandas as pd

from analyzer.metatypes import OpType


def verify_circle_free(dep_on, comm_groups, op2gid):
    super_node = {}
    for node in dep_on:
        if node in op2gid:
            super_node[node] = str(comm_groups[op2gid[node]])
        else:
            super_node[node] = str(node)

    super_son = {}
    for v in dep_on:
        super_son.setdefault(super_node[v], set())
        for u in dep_on[v]:
            # u->v
            super_son.setdefault(super_node[u], set()).add(super_node[v])

    visited = set()
    stack = []
    cir = []

    def dfs(u):  # tarjan-like dfs to find circles
        for v in super_son[u]:
            if v in stack:  # going backward to its ancester
                cir.extend(stack[stack.index(v):])
                cir.append(v)
                return True
            if v in visited:  # going to another branch
                continue
            stack.append(v)
            visited.add(v)
            if dfs(v):
                return True
            stack.pop()
        return False

    super_source = 'start'
    super_son[super_source] = set()
    for u in dep_on:
        if len(dep_on[u]) == 0:
            super_son[super_source].add(super_node[u])
    if dfs(super_source):
        assert False, f"Circle exists: {cir}. Potential bug in dependency modeling."
    assert len(visited) == len(tot:=set(super_node.values())), f"Graph not connected. Can only reach {len(visited)} nodes but total nodes={len(tot)}\n"\
    f"{visited=}. Non-visited={tot - visited}"


def _scatter_arr_back(op2src, op2idx, dst):
    for op, arr in op2src.items():
        dst[op2idx[op]] = arr


def compute_op2dfidx(df):
    return {
        df.op.iloc[group.index[0]]: group.index.to_numpy() for _, group in df[['opid']].groupby('opid', observed=False)
    }


def simulate(dep_on,
             comm_groups: List[List[str]],
             op2gid: Dict,
             need_dp_sync,
             df: pd.DataFrame,
             step_starts=None,
             inplace=False,
             op2dfidx=None,
             need_df=True,
             df_overwrite: pd.DataFrame = None) -> pd.DataFrame:
    """Simulates the timeline ('start_ts', 'end_ts', 'duration' for each op) with topological sort, 
    based on dependency model and the given op execution time in the 'duration' or 'transfer_duration' column.
    For performance reasons, we simulate in a "SIMD" way, where different DP ranks and steps are treated 
    as multiple data (MD) and processed together, since they share the same dependency model. 
    One special treatment is required for DP-level commnucation, where we need to ensure synchronization.

    Parameters:
        dep_on (Dict): A dictionary for in-rank dependencies.
        comm_groups (List[List[str]]): A list of communication groups for cross-rank dependencies, 
                                       where each group is a list of operation names.
        op2gid (Dict): A mapping from operation names to their corresponding communication group IDs.
        need_dp_sync (Set): A set of operations that require data-parallel synchronization.
        df (pd.DataFrame): A DataFrame containing the following columns:
            - 'dp_rank': Data-parallel rank of the operation.
            - 'step': Step number of the operation.
            - 'op': Operation name.
            - 'duration': Duration of the operation.
            - 'is_comm': Boolean indicating if the operation is a communication operation.
            - 'transfer_duration': Transfer duration of the communication operation.
            - 'launch_delay': After clearing its dependencies, each operation will still be delayed 
                by this amount of time before it starts. Only used to estimate simulation error;
                otherwise, it is 0.
            Additionally, for efficiency `df` should have its index sorted from 0 to n-1.

        step_starts (Optional[pd.Series]): A Series indicating the start times of each step.
                                           If not provided, it defaults to 0.
        inplace (bool): If True, modifies the input DataFrame in place. Defaults to False.
        op2dfidx (Optional[Dict]): A mapping from operation names to their corresponding DataFrame indices,
                                   for efficiency purposes. Defaults to None.
        need_df (bool): If True, ensures the returned DataFrame is valid. If False, only step times are computed. Defaults to True.
        df_overwrite (Optional[pd.DataFrame]): 
            A DataFrame used for zero-copy optimization to overwrite specific columns in `df`. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - The modified DataFrame with additional columns ('start_ts', 'end_ts', 'duration') if `need_df` is True, otherwise None.
            - A DataFrame containing step times with columns:
                - 'step': Step number.
                - 'step_time': Time taken for each step.

    """
    if not inplace:
        df = df.copy()
    dp_size = df.dp_rank.max() + 1
    if need_df:
        df['start_ts'] = np.nan
        df['end_ts'] = np.nan
    # op2df = {name: group.copy() for name, group in df.groupby('op')}
    assert (df.index == np.arange(
        len(df))).all(), f"df's index ({df.index}) needs to be [0,1,2,...,n-1] for performance purposes!"
    if op2dfidx is None:
        op2dfidx = compute_op2dfidx(df)
    op2start_ts = {}
    op2end_ts = {}
    is_comm_arr = df.is_comm.to_numpy()
    step_arr = df.step.to_numpy()
    duration_arr = (df_overwrite
                    if df_overwrite is not None and 'duration' in df_overwrite.columns else df).duration.to_numpy()
    transfer_duration_arr = (df_overwrite if df_overwrite is not None and 'transfer_duration' in df_overwrite.columns
                             else df).transfer_duration.to_numpy()
    launch_delay_arr = (df_overwrite if df_overwrite is not None and 'launch_delay' in df_overwrite.columns else
                        df).launch_delay.to_numpy()
    start_ts_arr = df.start_ts.to_numpy()
    end_ts_arr = df.end_ts.to_numpy()

    finished_ops = set()
    started_ops = set()
    group_started_cnt = [0 for _ in comm_groups]

    # topsort
    deg = {}
    nxt = {}
    for v, deps in dep_on.items():
        deg[v] = len(deps)
        nxt.setdefault(v, [])
        for u in deps:
            # u->v
            nxt.setdefault(u, []).append(v)

    queue = deque([v for v in dep_on if deg[v] == 0])
    opt_end_ts = None
    for _ in range(len(dep_on)):
        started = finished = None
        assert len(queue) > 0, f"queue empty after {_}/{len(dep_on)} iterations"
        v = queue.popleft()
        deps = dep_on[v]
        assert v not in started_ops, f"{v=}, {started_ops=}"
        v_is_comm = is_comm_arr[op2dfidx[v][0]]
        # u --> v
        assert len(deps) == 0 or all(
            u in finished_ops for u in deps), f"Queue front {v=} has some ops not finished, {deps=}"
        # launch op
        started = v
        if len(deps) == 0:  # first op. since step starts at zero, it starts at launch_delay
            op2start_ts[v] = launch_delay_arr[op2dfidx[v]] + (step_starts[step_arr[op2dfidx[v]]].to_numpy()
                                                              if step_starts is not None else 0)
        else:
            start_ts = op2end_ts[deps[0]]
            for u in deps[1:]:
                start_ts = np.maximum(start_ts, op2end_ts[u])
            op2start_ts[v] = start_ts + launch_delay_arr[op2dfidx[v]]
        if v_is_comm:
            group_started_cnt[op2gid[v]] += 1

        # finish op if it's a compute op or all the participants have started
        if not v_is_comm or group_started_cnt[op2gid[v]] == len(comm_groups[op2gid[v]]):
            finished = v

        assert started is not None, f"{_} vs {len(dep_on)}"
        started_ops.add(started)

        if finished is not None:
            new_finished = set([finished])
            is_comm = is_comm_arr[op2dfidx[finished][0]]
            if not is_comm:
                op2end_ts[finished] = op2start_ts[finished] + duration_arr[op2dfidx[finished]]
            else:  # set end_ts for the whole comm-group
                # find the max start_ts of the comm group
                global_start_ts = np.maximum.reduce([op2start_ts[v] for v in comm_groups[op2gid[finished]]
                                                    ])  # length: dp_rank*num_steps
                if finished in need_dp_sync:  # additionally reduce along dp to find the max start_ts
                    tmp = pd.DataFrame({'step': step_arr[op2dfidx[v]], 'start_ts': global_start_ts})
                    assert ((cnt := tmp.groupby('step')['start_ts'].count()) == dp_size).all(), f"{dp_size=} vs {cnt=}"
                    global_start_ts = tmp.groupby('step')['start_ts'].transform('max').to_numpy()
                for v in comm_groups[op2gid[finished]]:
                    op2end_ts[v] = global_start_ts + transfer_duration_arr[op2dfidx[v]]
                    # df.iloc[op2dfidx[v], ci['end_ts']] = op2end_ts[v]
                    new_finished.add(v)
            finished_ops |= new_finished
            for u in new_finished:
                for v in nxt[u]:
                    deg[v] -= 1
                    assert deg[v] >= 0, f"deg[{v}] is {deg[v]}<0, {u=}!"
                    if deg[v] == 0:
                        queue.append(v)

            # get step time
            v = finished
            if df.optype.iloc[op2dfidx[v][0]] == OpType.optimizer:  # end of step
                if opt_end_ts is None:
                    opt_end_ts = op2end_ts[v]
                    opt_step_arr = step_arr[op2dfidx[v]]
                else:
                    opt_end_ts = np.maximum(op2end_ts[v], opt_end_ts)
                    (opt_step_arr == step_arr[op2dfidx[v]]).all()

    assert opt_step_arr is not None, "optimzier should exist in the execution model as the final op."
    step_end_ts = pd.DataFrame({'step': opt_step_arr, 'end_ts': opt_end_ts}).groupby('step')['end_ts'].max()
    step_time = step_end_ts - (step_starts if step_starts is not None else 0)
    step_time = step_time.rename("step_time").reset_index()

    if need_df:
        _scatter_arr_back(op2start_ts, op2dfidx, start_ts_arr)
        _scatter_arr_back(op2end_ts, op2dfidx, end_ts_arr)
        df['start_ts'] = start_ts_arr
        df['end_ts'] = end_ts_arr

        assert len(started_ops) == len(
            dep_on), f'{len(started_ops)} vs {len(dep_on)}: {[i for i in dep_on if i not in started_ops]} not started.'
        assert len(finished_ops) == len(
            dep_on
        ), f'{len(finished_ops)} vs {len(dep_on)}: {[i for i in dep_on if i not in finished_ops]} not finished.'
        df_ret = df

        assert not df_ret[[
            'start_ts', 'end_ts'
        ]].isnull().any().any(), f"df has nans: {df_ret[df_ret[['start_ts', 'end_ts']].isnull().any(axis=1)]}"

        df_ret['duration'] = df_ret['end_ts'] - df_ret['start_ts']
        assert not df_ret[['start_ts', 'end_ts', 'launch_delay']].isna().any().any(), f"{df_ret=}"
        step_time.step_time = step_time.step_time.astype('float64')
    else:
        df_ret = None

    return df_ret, step_time


def simulate_if_replace(sim, index, col, df_ref, df, need_df=True):
    """Simulate the result if the ops of given index in `df` are replaced with `df_ref`.
    col: str, can be 'launch_delay', 'duration' or 'transfer_duration'."""

    col = ['duration', 'launch_delay', 'transfer_duration'] if col is None else col
    if 'duration' in col and 'transfer_duration' not in col:
        col.append('transfer_duration')
    if need_df:
        df_sim = df.copy()
        df_sim.loc[index, col] = df_ref.loc[index, col]
        return sim(df_sim, need_df=need_df)  # could be inplace
    else:  # "zero-copy" optimization, only copy the columns in `col``.
        df_col = df[col].copy()
        df_col.loc[index, col] = df_ref.loc[index, col]
        return sim(df, need_df=need_df, df_overwrite=df_col)  # could be inplace
