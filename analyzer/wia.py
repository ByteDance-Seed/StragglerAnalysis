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
import os
import pickle
import dataclasses
import functools
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
try:
    import ujson as json
except:
    import json

from analyzer.analysis_logger import AnalysisLogger
from analyzer.metatypes import JobMeta, AnalyzerResult, OpType
from analyzer.timers import RecordTime
from analyzer.dependency_model import setup_dependency
from analyzer.analysis_util import compress_dtype, get_step_time, get_step_time_ratios, set_op_id_, simulation_analysis_no_delay, simulation_analysis_reduce_dim
from analyzer.simulator import compute_op2dfidx, simulate, simulate_if_replace, verify_circle_free


def calc_launch_delay(df: pd.DataFrame, deps: Dict[str, List[str]]):
    """
    CPU-side operations like data-loading are not recorded and hence not reflected in the dependency model.
    Usually they are overlapped by GPU operations and do not affect the simulation.
    When the overlap is not perfect, neglecting them results in a gap before the next operation in the timeline.
    We call the gap `launch delay` and calculate it for each operation, and use them later to compute simulation errors.

    Formally, `launch_delay` is the delay between the start time of an operation and the maximum end time
    of its dependencies. If an operation has no dependencies, the delay is calculated relative
    to the global start time of the corresponding step.

    Parameters:
        df (pd.DataFrame): A DataFrame containing the following columns:
            - 'op': The name of the operation.
            - 'start_ts': The start timestamp of the operation.
            - 'end_ts': The end timestamp of the operation.
            - 'step': The step to which the operation belongs.
        deps (Dict[str, List[str]]): A dictionary mapping operation names to a list of their
            dependent operation names.

    Returns:
        pd.DataFrame: The modified DataFrame with an additional column `launch_delay`.
    """
    # NOTE: this is an inplace operation!

    df.reset_index(drop=True, inplace=True)
    df['launch_delay'] = np.nan

    op2dfidx = {name: group.index for name, group in df[['op']].groupby('op', observed=False)}

    start_ts_arr = df.start_ts.to_numpy()
    end_ts_arr = df.end_ts.to_numpy()
    step_arr = df.step.to_numpy()
    launch_delay_arr = df.launch_delay.to_numpy()

    step_start = df.groupby('step')['start_ts'].min()
    for tv in op2dfidx:
        end_tss = [end_ts_arr[op2dfidx[tu]] for tu in deps[tv]]
        if len(end_tss) == 0:
            end_ts = step_start[step_arr[op2dfidx[tv]]].to_numpy()
        else:
            end_ts = end_tss[0]
            for end_ts_i in end_tss[1:]:
                end_ts = np.maximum(end_ts, end_ts_i)
        launch_delay_arr[op2dfidx[tv]] = start_ts_arr[op2dfidx[tv]] - end_ts
    EPS = 1e-2  # 10ms
    if not (df.launch_delay > -EPS).all():
        AnalysisLogger().warning(
            f"launch delay has negative values <= {-EPS}: {df[~(df.launch_delay > -EPS)][['step', 'dp_rank', 'stage', 'op', 'launch_delay']]}"
        )
        AnalysisLogger().warning("Setting to zero.")
        df.loc[df.launch_delay <= -EPS, 'launch_delay'] = 0
    return df


def fix_negative_transfer_duration(df: pd.DataFrame, EPS, msg):
    assert df[df.transfer_duration <= -EPS].optype.isin([OpType.forward_send, OpType.backward_send]).all(),\
        f"{msg}\nFixing failed. Only support pp-send ops being the negative (happening before peer's) but got `{df[df.transfer_duration <= -EPS].optype.unique()}`"
    df.loc[df.transfer_duration <= -EPS, 'transfer_duration'] = 0


def get_transfer_time(df, op2gid, need_dp_sync):
    comm_ops = set(list(op2gid.keys()))
    df = df.copy()
    df_comm = df[df.op.isin(comm_ops)].copy()
    dp_size = df_comm.dp_rank.max() + 1
    df_comm['comm_gid'] = df_comm['op'].map(op2gid) * (dp_size + 1) + (~df_comm['op'].isin(need_dp_sync)) * (
        1 + df_comm['dp_rank'].astype('int64'))

    g = df_comm.groupby(['step', 'comm_gid'])
    df_comm['start_ts_max'] = g.start_ts.transform('max')
    df_comm['end_ts_min'] = g.end_ts.transform('min')

    # process_comm
    df_comm['transfer_duration'] = df_comm['end_ts'] - df_comm['start_ts_max']
    EPS = 1e-2  # 10ms
    df_comm.loc[df_comm.transfer_duration <= -EPS, 'transfer_duration'] = 0
    if (df_comm.transfer_duration <= -EPS).any():
        msg = f"negative transfer_duration found:\n{df_comm[df_comm.transfer_duration <= -EPS][['step', 'rank', 'op', 'transfer_duration', 'duration']]}\nTrying to fill with 0..."
        AnalysisLogger().warning(msg)
        fix_negative_transfer_duration(df_comm, EPS, msg)

    df['is_comm'] = False
    df_comm['is_comm'] = True
    df.loc[df_comm.index, ['is_comm', 'transfer_duration']] = df_comm

    tmp = df[df.step.eq(df.step.min()) & df.op.isin(
        [OpType.embedding_grads_all_reduce + "_0_0", OpType.embedding_grads_all_reduce + f"_0_{df.stage.max()}"])]
    assert np.allclose(tmp.transfer_duration.to_numpy(),
                       (tmp.end_ts - tmp.groupby('dp_rank').start_ts.transform('max')
                       ).to_numpy()), "embedding_grads_all_reduce transfer_duration check failed"

    return df


def get_op_name(*, op_seq_id=None, stage=None, optype=None, seq_id=None):
    if optype is not None and seq_id is not None:
        op_seq_id = get_optype_seq_id_name(optype, seq_id)
    if op_seq_id is not None:
        assert seq_id is None and optype is None and stage is not None
        is_series = isinstance(op_seq_id, pd.Series)
        if is_series:
            assert isinstance(
                op_seq_id.dtype,
                pd.CategoricalDtype), f"Expected to be categorical when passing in a Series, but got {op_seq_id.dtype}"
            op_seq_id = op_seq_id.astype(str)
        ret = op_seq_id + '_' + stage
        if is_series:
            ret = ret.astype("category")
        return ret
    raise ValueError("Unknown combination for get_op_name")


def get_optype_seq_id_name(optype, seq_id):
    is_series = isinstance(optype, pd.Series)
    if isinstance(seq_id, pd.Series):
        seq_id = seq_id.astype(str)
    elif isinstance(seq_id, int):
        seq_id = str(seq_id)
    else:
        raise ValueError(f"Unsupported type of seq_id: {type(seq_id)}")
    if is_series:
        assert isinstance(
            optype.dtype,
            pd.CategoricalDtype), f"Expected to be categorical when passing in a Series, but got {optype.dtype}"
        optype = optype.astype(str)
    ret = optype + '_' + seq_id
    if is_series:
        ret = ret.astype("category")
    return ret


def parse_trace(job_meta: JobMeta, trace, verbose=False):
    df = pd.read_parquet(trace)
    required_cols = {
        'dp_rank', 'stage', 'rank', 'step', 'optype', 'start_ts', 'duration', 'seq_id', 'mc', 'mb_id', 'gmc'
    }
    assert required_cols.issubset(df.columns), f"Missing columns in trace file: {required_cols - set(df.columns)}"
    df['end_ts'] = df['start_ts'] + df['duration']

    df['mc'] = df['mc'].astype('Int64')
    # setup op name: <optype>_<seq_id>_<stage>
    df['optype_seq_id'] = get_optype_seq_id_name(df['optype'], df['seq_id'])
    df['op'] = get_op_name(op_seq_id=df['optype_seq_id'], stage=df['stage'].astype(str))

    with RecordTime('setup_dependency', print_res=verbose):
        dep_on, comm_groups, op2gid, need_dp_sync = setup_dependency(df, job_meta.dsp_size)
    df['launch_delay'] = 0
    df = calc_launch_delay(df, dep_on)  # optinal and can be skipped. only used for simulation error esitmation.
    df = get_transfer_time(df, op2gid, need_dp_sync)
    df.reset_index(drop=True, inplace=True)
    set_op_id_(df)
    verify_circle_free(dep_on, comm_groups, op2gid)
    if not np.allclose((t1 := get_step_time(df)).to_numpy(),
                       (t2 := get_step_time(simulate(dep_on, comm_groups, op2gid, need_dp_sync, df)[0])).to_numpy()):
        AnalysisLogger().info(
            f"Warning: simulation on original trace doesn't yield the same result. Potential inaccuracy here.\nOriginal={t1}\nvs\nSimulation={t2}\nAvg: {t1.mean()} vs {t2.mean()}"
        )
    df = df.sort_values(by=['step', 'rank', 'optype', 'seq_id', 'start_ts']).reset_index(drop=True)
    df = compress_dtype(df, compress_start_end_ts=True)
    assert all(isinstance(df[c].dtype, pd.CategoricalDtype) for c in ['optype', 'optype_seq_id', 'op']) , \
        f"String columns should've been casted to category dtype, but got {df.op.dtype=}, {df.optype.dtype=}, {df.optype_seq_id.dtype=}"
    if 'first-op' not in df.optype.cat.categories:  # for later launch_decompose calculation
        df.optype = df.optype.cat.add_categories(['first-op'])
    return dep_on, comm_groups, op2gid, need_dp_sync, df


def decompose_by_opgroup(sim: Callable[[pd.DataFrame], pd.DataFrame],
                         df: pd.DataFrame,
                         df_ideal: pd.DataFrame,
                         step_t_ideal: pd.DataFrame,
                         target_col: str,
                         op_group: pd.Series,
                         target_steps: List[int],
                         verbose=False) -> Tuple[Dict[Any, float], Dict[int, Dict[Any, float]]]:
    """The use case is to find the major contributor to the slowdown from df_ideal to df. It can be useful to
     attribute the slowdown to different op groups. It enumerates each op group and estimate the slowdown when 
     `df_ideal` have the `target_col` (either duration or launch_delay) of the group replaced with `df`.
     
     Parameters:
        sim: the simulator function
        df: the original dataframe
        df_ideal: the ideal dataframe
        step_t_ideal: the ideal step time
        target_col: the column to be replaced
        op_group: providing a label for each op for grouping
        target_steps: the steps to be analyzed
        verbose: whether to show the progress bar
    Returns:
        res: the slowdown ratio for each op group. format: {<op group>: <overall slowdown ratio>}
        res_per_step: the slowdown ratio for each op group per step. format: 
            {<step>: {<op group>: <per-step slowdown ratio>}}
    """

    res, res_per_step = {}, {}

    iters = op_group.groupby(op_group)
    if verbose:
        iters = tqdm(iters)
    for group_name, group in iters:
        if verbose:
            iters.set_description(str(group_name))

        _, step_t_sim = simulate_if_replace(sim, group.index, target_col, df_ref=df, df=df_ideal, need_df=False)
        # per-step slowdown and overall slowdown when changing the given op_group from df_ideal to original df
        step_ratio, ratio = get_step_time_ratios(step_time_original=step_t_sim, step_time_new=step_t_ideal)

        res[group_name] = ratio
        for s in target_steps:
            # e.g., when op_group is by dp_rank, the res_per_step will be
            #   {<steps>:
            #     {<dp_rank>: slowdown for <dp_rank>}
            #     for <steps>
            #   }
            res_per_step.setdefault(s, {})[group_name] = step_ratio[s]

    def sort_last_level_dict(d: Dict):
        if not isinstance(list(d.values())[0], Dict):
            return dict(sorted(d.items(), key=lambda x: -x[1]))
        return {k: sort_last_level_dict(v) for k, v in d.items()}

    return sort_last_level_dict(res), sort_last_level_dict(res_per_step)


class WhatIfAnalyzer():

    def __init__(
        self,
        trace: str,
        root: Optional[str],
        steps: Optional[List[int]],
        decompose_threshold: float = 1.03,
        verbose: bool = False,
    ) -> None:
        """decompose_threshold: only decompose those with overall slowdown ratio >= decompose_threshold."""
        self.root = root
        self.steps = steps
        self.decompose_threshold = decompose_threshold
        self.verbose = verbose
        self.trace = trace

    def work(self, job_meta: JobMeta, trace: str):
        trial_name, run_id = job_meta.trial_name, job_meta.run_id
        AnalysisLogger().info(f'Working on {trial_name=}, {run_id=}')

        with RecordTime('Trace preprocessing', print_res=self.verbose):
            dep_on, comm_groups, op2gid, need_dp_sync, df = \
                parse_trace(job_meta, trace, verbose=self.verbose)
            dependency_model = (dep_on, comm_groups, op2gid, need_dp_sync)

        if self.steps is not None:  # analyze only the specified steps
            steps = set(self.steps) & set(df.step.unique())
            assert len(steps) > 0, \
                f"Trace has no valid step in the given range. Please choose from steps in the trace: {df.step.unique()}"
            df = df[df.step.isin(steps)].reset_index()
            AnalysisLogger().info(f"Analyzing only these steps: {steps}")
            self.steps = steps

        # dump the processed trace and dependency model, for potential customized what-if analysis
        with RecordTime(name='Dump trace and dependency model', print_res=self.verbose):
            trial = job_meta.trial_name
            run_id = job_meta.run_id
            if self.root is not None:
                path = f'{self.root}/dataframe'
                os.makedirs(path, exist_ok=True)
                df.to_parquet(f'{self.root}/dataframe/{trial}-{run_id}-base.parquet')
                with open(f'{self.root}/dataframe/{trial}-{run_id}-attrs.pkl', 'wb') as fout:
                    pickle.dump(df.attrs, fout)  # contains the step_start dict

                os.makedirs(f'{self.root}/dependency_model', exist_ok=True)
                with open(f'{self.root}/dependency_model/{trial}-{run_id}.pkl', 'wb') as fout:
                    pickle.dump(dependency_model, fout)

        df.drop(columns=['mc', 'mb_id', 'gmc', 'seq_id'], inplace=True)

        op2dfidx = compute_op2dfidx(df)
        sim_func = functools.partial(simulate,
                                     dep_on,
                                     comm_groups,
                                     op2gid,
                                     need_dp_sync,
                                     step_starts=df.groupby('step')['start_ts'].min(),
                                     inplace=True,
                                     op2dfidx=op2dfidx)

        gc.collect()
        with RecordTime(name="top-level simulation", print_res=self.verbose):
            agg_kwargs = {'agg_method': 'mean', 'version': 'latest'}
            step_ratio_no_delay, ratio_no_delay, df_no_delay, step_t_no_delay = simulation_analysis_no_delay(
                sim_func, df)
            step_ratio_no_blocking, ratio_no_blocking, df_no_blocking, step_t_no_blocking = simulation_analysis_reduce_dim(
                sim_func, df_no_delay, ['dp_rank', 'stage', 'step'], **agg_kwargs)

            with RecordTime(name="get step time", print_res=self.verbose):
                step_t_res = {
                    'step_t_base': get_step_time(df).set_index('step').step_time.to_dict(),
                    'step_t_nodelay': get_step_time(df_no_delay).set_index('step').step_time.to_dict(),
                    'step_t_noblk': get_step_time(df_no_blocking).set_index('step').step_time.to_dict(),
                }

        gc.collect()

        if self.root is not None:  # dump trace for the simulated ideal
            os.makedirs(f'{self.root}/dataframe', exist_ok=True)
            df.to_parquet(f'{self.root}/dataframe/{trial}-{run_id}-base.parquet')
            df_no_delay.to_parquet(f'{self.root}/dataframe/{trial}-{run_id}-no-delay.parquet')
            df_no_blocking.to_parquet(f'{self.root}/dataframe/{trial}-{run_id}-no-blocking.parquet')

        optype2v2 = lambda x: {
            OpType.forward_recv: 'forward-pp-comm',
            OpType.forward_send: 'forward-pp-comm',
            OpType.backward_recv: 'backward-pp-comm',
            OpType.backward_send: 'backward-pp-comm'
        }.get(x, x)
        df['optype_backup'] = df['optype']
        df['optype'] = df['optype'].map(optype2v2)
        default_groupby = ['optype', 'dp_rank', 'stage']
        decompose_res = []
        with RecordTime(name="decomposition", print_res=self.verbose):
            work_items = [
                ('launch_decompose', df_no_delay, step_t_no_delay, ['launch_delay'], step_ratio_no_delay,
                 ratio_no_delay, default_groupby),
                ('blocking_decompose', df_no_blocking, step_t_no_blocking, ['duration'], step_ratio_no_blocking,
                 ratio_no_blocking, default_groupby),
            ]

            for method_name, df_ideal, step_t_ideal, target, step_ratio, ratio, groupby in work_items:
                if self.verbose:
                    AnalysisLogger().info(f'---> Working on {method_name}')
                if ratio < self.decompose_threshold:
                    continue
                if method_name == 'launch_decompose':
                    is_first_fwd = df.stage.eq(0) & df.op.str.startswith(OpType.forward_compute + '_0')
                    is_first_fwd_recv = (df.stage > 0) & df.op.str.startswith(OpType.forward_recv + '_0')
                    df.loc[is_first_fwd | is_first_fwd_recv, 'optype'] = 'first-op'

                for col in groupby:
                    cur_res = decompose_by_opgroup(sim=sim_func,
                                                   df=df,
                                                   df_ideal=df_ideal,
                                                   step_t_ideal=step_t_ideal,
                                                   target_col=target,
                                                   op_group=df[col],
                                                   target_steps=self.steps or list(step_ratio.keys()),
                                                   verbose=self.verbose)

                    decompose_res.append((f"{method_name}_by_{col}", cur_res[0]))
                    decompose_res.append((f"{method_name}_by_{col}_per_step", cur_res[1]))
                if method_name == 'launch_decompose':
                    df.loc[is_first_fwd, 'optype'] = optype2v2(OpType.forward_compute)
                    if is_first_fwd_recv.any():
                        df.loc[is_first_fwd_recv, 'optype'] = optype2v2(OpType.forward_recv)

            decompose_res = dict(decompose_res)

        named_res = step_t_res.copy()
        named_res.update(decompose_res)
        AnalysisLogger().info(f'Analysis success for {trial=}, {run_id=}')
        return step_ratio_no_delay, step_ratio_no_blocking, ratio_no_delay, ratio_no_blocking, named_res

    def analyze(self, job_meta: JobMeta) -> AnalyzerResult:
        RecordTime.timers.clear()
        with RecordTime(name="e2e", print_res=self.verbose):
            (
                launch_step_slowdown,
                blocking_step_slowdown,
                launch_slowdown,
                blocking_slowdown,
                named_res,
            ) = self.work(job_meta, self.trace)

        analyze_res = AnalyzerResult(
            job_meta=job_meta,
            launch_slowdown=launch_slowdown,
            blocking_slowdown=blocking_slowdown,
            launch_step_slowdown=launch_step_slowdown,
            blocking_step_slowdown=blocking_step_slowdown,
            **named_res,
        )
        return analyze_res


def parse_step(step_str: Optional[List[str]]) -> Optional[List[int]]:
    if step_str is None:
        return None
    steps = set()
    for s in step_str:
        if '-' not in s:
            steps.add(int(s))
        else:
            start_str, end_str = s.split('-')
            steps.update(range(int(start_str), int(end_str) + 1))
    return list(steps)


def main():
    parser = argparse.ArgumentParser('What-If Analysis')
    parser.add_argument('--root', help='root directory to dump the processed trace and dependency model')
    parser.add_argument('--trace', help='load the trace (without dependency model) from the given path',
                        required=True)  # consider merge into metadata
    parser.add_argument(
        '--step',
        nargs='*',
        type=str,
        help="Specific steps to perform the analysis on. E.g., --step 12 '20-23' includes 12,20,21,22,23")
    parser.add_argument('--decompose-thres', type=float, default=1.03)
    parser.add_argument('--dump-dir', default='./sim-result', help='directory to dump the analysis result')
    parser.add_argument('--metadata', help='path to a yaml file containing metadata for the trial', required=True)
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    args = parser.parse_args()
    assert args.metadata.endswith(".yaml") or args.metadata.endswith(
        ".yml"), f"unsupported metadata file {args.metadata}"
    with open(args.metadata, 'r') as f:
        job_meta = JobMeta(**yaml.safe_load(f))

    res = WhatIfAnalyzer(args.trace,
                         root=args.root,
                         steps=parse_step(args.step),
                         decompose_threshold=args.decompose_thres,
                         verbose=args.verbose).analyze(job_meta)

    dump_dir = args.dump_dir
    os.makedirs(f'{dump_dir}', exist_ok=True)
    output_dir = f"{dump_dir}/{job_meta.trial_name}.json"
    with open(output_dir, "w") as f:
        json.dump(dataclasses.asdict(res), f, indent=2)

    AnalysisLogger().info(f"Result dumped to {dump_dir}/{job_meta.trial_name}.json")


if __name__ == '__main__':
    main()
