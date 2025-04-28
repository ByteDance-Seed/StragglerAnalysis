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

from analyzer.analysis_logger import AnalysisLogger
from analyzer.explain_slowdown import explain_slowdown
from analyzer.analysis_util import simulation_analysis_no_delay, simulation_analysis_reduce_dim
from analyzer.simulator import simulate


def classify_slow_last_stage(df_base, df_nodelay, df_noblocking, sim_func):

    def row_select_func(df):
        pred = df.stage == df.stage.max()
        return pred.index.to_numpy()[np.nonzero(pred.to_numpy())]

    indexes = row_select_func(df_base)
    assert np.all(indexes == row_select_func(df_nodelay)), f"{indexes=} vs {row_select_func(df_nodelay)=}"
    assert np.all(indexes == row_select_func(df_noblocking)), f"{indexes=} vs {row_select_func(df_noblocking)=}"
    return explain_slowdown(df_nodelay, df_noblocking, indexes, ['duration', 'transfer_duration'], sim_func)


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

    res_long_loss = classify_slow_last_stage(df_base, df_nodelay, df_noblk, sim_func)

    return res_long_loss, trial_id, run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-result')
    parser.add_argument('--trial')
    parser.add_argument('--root')
    parser.add_argument('--dump-dir', default='./sim-result')
    args = parser.parse_args()

    with open(args.analysis_result) as f:
        ret = work(args.root, args.trial, json.load(f))
    Ms = ret[0]['rem']
    ret[0]['Ms'] = Ms
    output = f'{args.dump_dir}/ms-{args.trial}.json'

    json.dump(ret[0], open(output, 'w'), indent=4)
    AnalysisLogger().info(f'Ms={Ms*100:.1f}%')
    AnalysisLogger().info(f'Detailed result saved to {output}')


if __name__ == '__main__':
    main()
