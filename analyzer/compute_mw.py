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

import argparse
import functools
import json
import pickle
import os

import pandas as pd
import numpy as np

from analyzer.explain_slowdown import explain_slowdown
from analyzer.analysis_util import simulation_analysis_no_delay, simulation_analysis_reduce_dim
from analyzer.simulator import simulate
from analyzer.analysis_logger import AnalysisLogger

RATIO = 3


def cdiv(a, b):
    return (a + b - 1) // b


def compute_for_ratio(borl, dim, df_original, df_ideal, replace_cols, row, sim_func):
    ratio = RATIO

    def row_select_func(df):
        scores = row[f'pp_{borl}_decompose_by_{dim}']
        # get keys of top ratio items
        n = cdiv(len(scores) * ratio, 100)
        keys = sorted(scores.keys(), key=lambda k: (scores[k], -int(k.split(',')[0])), reverse=True)[:n]
        # translate keys nto dp_rank or stage
        ranks = [int(k.split(',')[0]) for k in keys]
        pred = df[dim].isin(ranks)
        ret = pred.index.to_numpy()[np.nonzero(pred.to_numpy())]
        return ret

    indexes = row_select_func(df_original)
    assert np.all(indexes == row_select_func(df_ideal)), f"{indexes=} vs {row_select_func(df_ideal)=}"
    ret = explain_slowdown(df_original, df_ideal, indexes, replace_cols, sim_func)
    return ret


def get_rank_scores(row, ty, df_base):
    # get rank to dp_rank and stage mapping
    df_base = df_base[['dp_rank', 'stage', 'rank']]
    df_base = df_base.drop_duplicates()
    df_base = df_base.set_index('rank')

    decompose_by_dp_rank = row[f'{ty}_decompose_by_dp_rank']
    decompose_by_stage = row[f'{ty}_decompose_by_stage']

    if pd.isnull(decompose_by_dp_rank):
        return

    a = np.zeros((len(decompose_by_dp_rank), len(decompose_by_stage)))

    for dp_rank_key in decompose_by_dp_rank:
        dp_rank = int(dp_rank_key.split(',')[0])
        for stage_key in decompose_by_stage:
            pp_rank = int(stage_key.split(',')[0])
            a[dp_rank, pp_rank] = min(decompose_by_dp_rank[dp_rank_key], decompose_by_stage[stage_key])
    scores = {f'{rank},{rank}': a[df_base.loc[rank, 'dp_rank'], df_base.loc[rank, 'stage']] for rank in df_base.index}
    return scores


def classify_machine_problem(df_base, df_nodelay, df_allnodelay, df_noblocking, row, sim_func):
    borl = 'blocking'
    row[f'pp_{borl}_decompose_by_rank'] = get_rank_scores(row, borl, df_base)
    df_original, df_ideal = df_allnodelay, df_noblocking
    replace_cols = ['duration', 'transfer_duration']
    dim = 'rank'
    return compute_for_ratio(borl, dim, df_original, df_ideal, replace_cols, row, sim_func)


def work(root, trial_id, row):
    run_id = 0
    dep_model = f'{root}/dependency_model/{trial_id}-{run_id}.pkl'  # dependency model
    base_path = f'{root}/dataframe/{trial_id}-{run_id}-base.parquet'  # original trace
    assert os.path.exists(base_path), f'{base_path=} not found'
    assert os.path.exists(dep_model), f'{dep_model=} not found'

    dep_on, comm_groups, op2gid, need_dp_sync = pickle.load(open(dep_model, 'rb'))
    sim_func = functools.partial(simulate, dep_on, comm_groups, op2gid, need_dp_sync,
                                 inplace=True)  #, step_starts=df_base.groupby('step')['start_ts'].min(), inplace=True)

    df_base = pd.read_parquet(base_path)

    df_base['end_ts'] = df_base['start_ts'] + df_base['duration']
    step_ratio_no_delay, ratio_no_delay, df_nodelay, step_t_no_delay = simulation_analysis_no_delay(sim_func, df_base)
    step_ratio_no_blocking, ratio_no_blocking, df_noblk, step_t_no_blocking = simulation_analysis_reduce_dim(
        sim_func, df_nodelay, ['dp_rank', 'stage', 'step'], version='v4')

    res_fixed_location = classify_machine_problem(df_base, df_nodelay, df_nodelay, df_noblk, row, sim_func)

    return res_fixed_location, trial_id, run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-result')
    parser.add_argument('--trial')
    parser.add_argument('--root')
    parser.add_argument('--dump-dir', default='./sim-result')
    args = parser.parse_args()

    ret = work(args.root, args.trial, json.load(open(args.analysis_result)))
    Mw = ret[0]['rem']
    ret[0]['Mw'] = Mw
    output = f'{args.dump_dir}/mw-{args.trial}.json'
    json.dump(ret[0], open(output, 'w'), indent=4)
    AnalysisLogger().info(f'Mw={Mw*100:.1f}%. Detailed result saved to {output}')


if __name__ == '__main__':
    main()
