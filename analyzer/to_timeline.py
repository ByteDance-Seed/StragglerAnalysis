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

import gzip
import os
import pandas as pd
import argparse
try:
    import ujson as json
except:
    import json


def _to_timeline(df: pd.DataFrame, step_st: int, step_ed: int, ranks: list[int]):
    df = df[(df['step'] >= step_st) & (df['step'] <= step_ed) & (df['rank'].isin(ranks))].copy()
    optype_types = {
        'compute': [
            'forward-compute', 'backward-compute', 'data-load', 'data-sync', 'optimizer', 'gc',
            'backward-wgrad-compute', 'forward-backward'
        ],
        'pp-comm': [
            'forward-send', 'backward-send', 'forward-recv', 'backward-recv', 'forward-send-backward-recv',
            'backward-send-forward-recv'
        ],
        'tp-comm': ['tp-reducescatter', 'tp-allgather', 'tp-allreduce'],
        'dp-comm': [
            'grads-reduce-scatter', 'separate-grads-all-reduce', 'grads-reduce-scatter-nonoverlapping',
            'params-all-gather', 'params-all-gather-nonoverlapping', 'embedding-grads-all-reduce',
            'layernorm-grads-all-reduce', 'optimizer-clip-main-grad', 'optimizer-inner-step'
        ],
        'misc': []
    }
    all_known_optypes = sum(optype_types.values(), [])
    unknown_optypes = []
    for m in df.optype.unique():
        if m not in all_known_optypes:
            unknown_optypes.append(m)
    # assert len(unknown_optypes) == 0, f'Unknown optypes: {unknown_optypes}'
    if len(unknown_optypes) > 0:
        print('Warning, unknown optypes found:', unknown_optypes)

    def get_tid(rank: int, optype: str):
        for i, (k, v) in enumerate(optype_types.items()):
            if k == 'misc' or optype in v:
                return rank * len(optype_types) + i
        # raise ValueError(f'Unknown optype: {optype}')
        raise RuntimeError("Should not reach here.")

    events = []
    tp_size = df['tp_rank'].nunique()
    pp_size = df['stage'].nunique()
    dp_size = df['dp_rank'].nunique()

    pp_ranks = df['stage'].unique()
    dp_ranks = df['dp_rank'].unique()
    if 'mb_id' not in df.columns:
        df['mb_id'] = None
    if 'model_chunk' not in df.columns:
        if 'mc' in df.columns:
            df['model_chunk'] = df.mc
        else:
            df['model_chunk'] = None

    df['seq_id'] = df['optype_seq_id'].map(lambda x: x.split('_')[-1]).astype(int)
    for optype, dp_rank, rank, start_ts, duration, step, model_chunk, mb_id, seq_id, transfer_duration in zip(
            df['optype'], df['dp_rank'], df['rank'], df['start_ts'], df['duration'], df['step'], df['model_chunk'],
            df['mb_id'], df['seq_id'], df['transfer_duration']):
        if not pd.isnull(transfer_duration):
            start_ts += duration - transfer_duration
            duration = transfer_duration
        evt = {
            'name': optype,
            'cat': optype,
            'pid': int(dp_rank),
            'tid': get_tid(rank, optype),
            'ts': start_ts * 1e6,  # us
            'dur': duration * 1e6,  # us
            'ph': 'X',
            'args': {
                'step': int(step),
                'model_chunk': int(model_chunk) if not pd.isnull(model_chunk) else None,
                'mb_id': int(mb_id) if not pd.isnull(mb_id) else None,
                'seq_id': int(seq_id) if not pd.isnull(seq_id) else None,
            },
        }
        events.append(evt)

    metadata_events = []
    for rank in ranks:
        pp_rank = int(df[df['rank'] == rank]['stage'].iloc[0])
        dp_rank = int(df[df['rank'] == rank]['dp_rank'].iloc[0])
        tp_rank = int(df[df['rank'] == rank]['tp_rank'].iloc[0])
        rank = int(rank)
        for i, mt in enumerate(optype_types.keys()):
            metadata_events.append({
                'name': 'thread_name',
                'pid': dp_rank,
                'tid': rank * len(optype_types) + i,
                'ph': 'M',
                'args': {
                    'name': f'PP rank {pp_rank}, TP rank {tp_rank}, {mt}',
                },
            })
            metadata_events.append({
                'name': 'thread_sort_index',
                'pid': dp_rank,
                'tid': rank * len(optype_types) + i,
                'ph': 'M',
                'args': {
                    'sort_index': rank * len(optype_types) + i,
                },
            })
    for dp_rank in dp_ranks:
        metadata_events.append({
            'name': 'process_name',
            'pid': int(dp_rank),
            'ph': 'M',
            'args': {
                'name': f"DP rank [{dp_rank}]",
            },
        })
    events += metadata_events
    return events


def to_timeline(df: pd.DataFrame, step_st: int = None, step_ed: int = None, rank3d: list[str] = None):
    rank3d = [':,:,:'] if rank3d is None else rank3d
    df = df.copy()

    if 'dp_rank' not in df.columns:
        df['dp_rank'] = 0
    if 'stage' not in df.columns:
        df['stage'] = 0
    if 'tp_rank' not in df.columns:
        df['tp_rank'] = 0
    if 'rank' not in df.columns:
        rank_cols = ['stage', 'dp_rank', 'tp_rank']
        df['rank'] = df.sort_values(rank_cols).groupby(rank_cols).ngroup()

    if step_st is None:
        step_st, step_ed = get_step_range(df)

    def parse(s: str):
        if s[0] == '[':
            assert s[-1] == ']'
            return list(map(int, s[1:-1].split(';')))
        return [int(s)]

    ranks = []
    for r3d in rank3d:
        dp, pp, tp = r3d.split(',')
        f = df['rank'].notna()
        if dp != ':':
            f = f & (df['dp_rank'].isin(parse(dp)))
        if pp != ':':
            f = f & (df['stage'].isin(parse(pp)))
        if tp != ':':
            f = f & (df['tp_rank'].isin(parse(tp)))
        ranks.extend(df[f]['rank'].unique())
    ranks = list(set(ranks))
    return _to_timeline(df, step_st, step_ed, ranks)


def get_step_range(df: pd.DataFrame, step_st: int = None, step_ed: int = None):
    if step_st is None:
        step_st = df['step'].min()
    if step_ed is None:
        step_ed = df['step'].min() + 10
    return step_st, step_ed


# def dump_timeline(df, file_name='', step_st=None, step_ed=None, rank3d=None):
#     from tools.to_timeline import get_step_range, to_timeline
#     import gzip
#     rank3d = [':,:,:'] if rank3d is None else rank3d
#     if step_st is None:
#         step_st, step_ed = get_step_range(df)


#     content = json.dumps(to_timeline(df, step_st, step_ed, rank3d)).encode('utf-8')
#     file = f'./timeline-{file_name}-{step_st}-{step_ed}-{rank3d}.json.gz'
#     with gzip.open(file, 'wb') as f:
#         f.write(content)
def apply_step_start(df, step_start):
    df['start_ts'] = df['start_ts'].astype('float64')
    df['end_ts'] = df['end_ts'].astype('float64')
    df['start_ts'] += df['step'].map(step_start)
    df['end_ts'] += df['step'].map(step_start)


def dump_timeline(filename, df, step_st, step_ed, rank3d=None, reset_step_start=False):
    if reset_step_start:
        step_start = df.groupby('step')['end_ts'].max() * 1.1
        step_start = step_start.sort_index().shift(1).fillna(0).cumsum()
        apply_step_start(df, step_start)
    content = json.dumps(to_timeline(df, step_st, step_ed, rank3d)).encode('utf-8')
    print('Dumping to', filename)
    with gzip.open(filename, 'wb') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description='Converts the raw logs to a timeline')
    parser.add_argument('input', type=str, help='Input datafrarme file')
    parser.add_argument('-o', type=str, help='Output directory', default='.')
    parser.add_argument('--step-st', type=int, help='Start step')
    parser.add_argument('--step-ed', type=int, help='End step. By default use first 10 steps')
    parser.add_argument('--rank3d',
                        type=str,
                        nargs='+',
                        default=[':,:,:'],
                        help='Rank 3D in the format of dp,pp,tp. Use : for wildcard')
    parser.add_argument('--reset-step-start', action='store_true')
    args = parser.parse_args()
    df = None
    for func in [pd.read_pickle, pd.read_parquet]:
        if df is not None:
            break
        try:
            df = func(args.input)
        except:
            pass
    assert df is not None, "Unknown input format. Currently only support pickle and parquet"
    if isinstance(df, tuple):  # just a HACK
        df = df[-1]
    if 'end_ts' not in df.columns:
        df['end_ts'] = df['start_ts'] + df['duration']
    step_st, step_ed = get_step_range(df, args.step_st, args.step_ed)
    dump_timeline(args.o + f'/timeline-{os.path.basename(args.input)}-{step_st}-{step_ed}-{args.rank3d}.json.gz',
                  df,
                  step_st,
                  step_ed,
                  args.rank3d,
                  reset_step_start=args.reset_step_start)


if __name__ == '__main__':
    main()
