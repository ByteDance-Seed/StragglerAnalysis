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
import json

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_heatmap(analysis_result: dict,
                 ty,
                 dp_ranks=None,
                 pp_ranks=None,
                 save=None,
                 cbar=True,
                 fmt='.2g',
                 vmin=1,
                 vmax=1.5):
    """Plot a heatmap to visualize the analysis result.

    Parameters:
        analysis_result (dict): The analysis result containing the data to plot. 
            It must include the keys 'dp_size', 'pp_size', 
            'pp_<ty>_decompose_by_dp_rank', and 'pp_<ty>_decompose_by_stage'.
        ty (str): The type of analysis result to plot. Must be one of ['blocking', 'overlap'].
        dp_ranks (list, optional): A list of DP ranks to include in the heatmap. 
            Defaults to all DP ranks.
        pp_ranks (list, optional): A list of PP ranks to include in the heatmap. 
            Defaults to all PP ranks.
        save (str, optional): The file path to save the heatmap image. 
            If None, the heatmap is not saved. Defaults to None.
        cbar (bool, optional): Whether to include a color bar in the heatmap. Defaults to True.
        fmt (str, optional): The string formatting for annotations in the heatmap. Defaults to '.2g'.
        vmin (float, optional): The minimum value for the heatmap color scale. Defaults to 1.
        vmax (float, optional): The maximum value for the heatmap color scale. Defaults to 1.5.

    Returns:
        None
    """
    if ty not in ['blocking', 'overlap']:
        raise ValueError(f'Invalid ty: {ty}, should be one of [blocking, overlap]')
    dp_size, pp_size = analysis_result['job_meta']['dp_size'], analysis_result['job_meta']['pp_size']
    if dp_ranks is None:
        dp_ranks = list(range(dp_size))
    if pp_ranks is None:
        pp_ranks = list(range(pp_size))
    decompose_by_dp_rank = analysis_result[f'{ty}_decompose_by_dp_rank']
    decompose_by_stage = analysis_result[f'{ty}_decompose_by_stage']

    assert not pd.isnull(decompose_by_dp_rank), 'No straggler detected in the given job'

    a = np.zeros((dp_size, pp_size))
    for dp_rank_key in decompose_by_dp_rank:
        #TODO(JK): fix the key format to only include dp rank
        dp_rank = int(dp_rank_key.split(',')[0])
        for stage_key in decompose_by_stage:
            pp_rank = int(stage_key.split(',')[0])
            a[dp_rank, pp_rank] = min(decompose_by_dp_rank[dp_rank_key], decompose_by_stage[stage_key])
    # matrix to dataframe
    a = pd.DataFrame(a.transpose()).loc[pp_ranks, dp_ranks]
    aspect_ratio = 3
    plt.subplots(figsize=(len(dp_ranks) / aspect_ratio, len(pp_ranks)))
    ax = sns.heatmap(
        a,
        linewidth=0.01,
        cmap='Reds',
        fmt=fmt,
        cbar_kws={
            'location': 'top',
            'aspect': 25,
            'fraction': 0.5,
        },
        annot=True,
        square=True,
        vmin=vmin,
        vmax=vmax,
        cbar=cbar,
    )
    ax.set_xlabel('DP rank')
    ax.set_ylabel('PP rank')
    if save:
        plt.savefig(save, bbox_inches='tight')
        print(f'Heatmap saved to {save}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-dir', default='./sim-result')
    parser.add_argument('--trial')
    parser.add_argument('--analysis-result')
    args = parser.parse_args()
    plot_heatmap(json.load(open(args.analysis_result)), 'blocking', save=f'{args.dump_dir}/heatmap-{args.trial}.png')
