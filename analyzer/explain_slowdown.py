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

from analyzer.analysis_util import get_step_time


def explain_slowdown(df_original, df_ideal, indexes, replace_cols, sim_func):
    """Explain the slowdown of the original timeline compared to the ideal timeline, by 
    estimating the effect of selected ops on their difference.
    Let `tideal` and `tori` be the average step time of the ideal and original timeline, respectively.
    Then the effect is estimated in two ways:
    1) addition: the effect of replacing the selected ops in the ideal timeline with orignal ones, 
       effectively "adding" back the straggling ops to ideal timeline. Let `tadd` be the resulted
        average step time. The effect is the extra step time `tadd - tori` it incurrs as a percentage 
        of the overall slowdown `tori - tideal`. Formally, `add = (tadd - tideal) / (tori - tideal)`.
    2) removal: the effect of replacing the selected ops in the original timeline with ideal ones,
       effectively "removing" the straggling ops from the original timeline. Let `trem` be the resulted
        average step time. The effect is `rem = (tori - trem) / (tori - tideal)`.
    ---*-------*------------------*-------*---->
       tideal   trem              tadd    tori
    """
    tori = get_step_time(df_original).step_time.mean()
    tideal = get_step_time(df_ideal).step_time.mean()

    df = df_original.copy()
    df.loc[indexes, replace_cols] = df_ideal.loc[indexes, replace_cols]
    trem = get_step_time(sim_func(df)[0]).step_time.mean()

    df = df_ideal.copy()
    df.loc[indexes, replace_cols] = df_original.loc[indexes, replace_cols]
    tadd = get_step_time(sim_func(df)[0]).step_time.mean()

    return {
        'trem': float(trem),
        'tadd': float(tadd),
        'tori': float(tori),
        'tideal': float(tideal),
        'rem': float((tori - trem) / (tori - tideal)),
        'add': float((tadd - tideal) / (tori - tideal))
    }
