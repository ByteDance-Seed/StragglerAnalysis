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

from typing import Dict

import pandas as pd

from analyzer.metatypes import OpType


def setup_chain_deps(deps, ops_chain, last_ops=None):
    last_ops = last_ops or []
    for op in ops_chain:
        deps.setdefault(op, []).extend(last_ops)
        last_ops = [op]


def get_sender_stage(f_or_b, send_or_recv, gmc, stage, vpp, pp):  # -> (sender stage, is_no_op)
    delta = {
        ('forward', 'send'): 1,
        ('forward', 'recv'): -1,
        ('backward', 'send'): -1,
        ('backward', 'recv'): 1,
    }
    peer_gmc = delta[(f_or_b, send_or_recv)] + gmc
    peer_stage = (delta[(f_or_b, send_or_recv)] + stage + pp) % pp
    tot_gmcs = vpp * pp
    is_no_op = peer_gmc >= tot_gmcs or peer_gmc < 0
    if send_or_recv == 'send':
        return stage, is_no_op
    else:
        return peer_stage, is_no_op


def setup_post_compute_dependency(deps, df_mp_group, ovlp_dp: bool, use_dp: bool):
    """Setup in-rank dependencies for ops after last bwd-compute."""
    pp = df_mp_group.stage.max() + 1
    for stage in range(pp):
        df_stage = df_mp_group[df_mp_group.stage.eq(stage)]
        df_comp = df_stage[df_stage.optype.isin([OpType.forward_compute, OpType.backward_compute
                                                ])].sort_values(['start_ts', 'seq_id', 'optype'])
        last_compute = df_comp.loc[df_comp.end_ts.idxmax()].op
        deps.setdefault(f'gc_0_{stage}', []).extend([last_compute])
        t1 = df_stage[df_stage.optype.eq(OpType.grads_reduce_scatter)].seq_id.max() + 1
        embedding = [f"{OpType.embedding_grads_all_reduce}_0_{stage}"] if stage == 0 or stage == pp - 1 else []
        if not use_dp:
            # bwd_compute -> gc -> layernorm -> embed -> clip -> opt
            setup_chain_deps(deps,
                             ops_chain=[
                                 f"{OpType.layernorm_grads_all_reduce}_0_{stage}", *embedding,
                                 f"{OpType.optimizer_clip_main_grad}_0_{stage}", f"{OpType.optimizer}_0_{stage}"
                             ],
                             last_ops=[f'gc_0_{stage}'])
        elif ovlp_dp:
            ##  ...   -> grads-rs_{n-1} -> layernorm -> embed -> sep-grads-rs -> clip -> opt
            setup_chain_deps(deps,
                             ops_chain=[
                                 f"{OpType.layernorm_grads_all_reduce}_0_{stage}", *embedding,
                                 f"{OpType.separate_grads_all_reduce}_0_{stage}",
                                 f"{OpType.optimizer_clip_main_grad}_0_{stage}", f"{OpType.optimizer}_0_{stage}"
                             ],
                             last_ops=[OpType.grads_reduce_scatter + f'_{t1 - 1}_{stage}', f'gc_0_{stage}'])
        else:
            # ... -> bwd_compute -> gc -> layernorm -> embed -> grads-rs -> ... instead
            setup_chain_deps(deps,
                             ops_chain=[
                                 f"{OpType.layernorm_grads_all_reduce}_0_{stage}", *embedding,
                                 *[f"{OpType.grads_reduce_scatter}_{i}_{stage}" for i in range(t1)],
                                 f"{OpType.optimizer_clip_main_grad}_0_{stage}", f"{OpType.optimizer}_0_{stage}"
                             ],
                             last_ops=[f'gc_0_{stage}'])


def setup_params_ag_deps(deps, df_mp_group):
    """
    Make 1st fwd-compute depend on params-all-gather of the same model-chunk.
    .. 
        ... --> fwd-compute_{i_0} -->  ...   --> fwd-compute_{i_{vpp-1}}
                ^                                      ^
               /                                      /  
        params-ag_{i_0} --> ... --> params-ag_{i_{vpp-1}}

    , where `vpp` is the number of model-chunks in the pipeline parallelism.
    """
    for stage in range(df_mp_group.stage.max() + 1):
        df_stage = df_mp_group[df_mp_group.stage.eq(stage)]
        ag_cnt = len(df_stage[df_stage.optype.eq(OpType.params_all_gather)])
        df_comp = df_stage[df_stage.optype.isin([OpType.forward_compute, OpType.backward_compute
                                                ])].sort_values(['start_ts', 'seq_id', 'optype'])
        mc_assigned = set()
        for i in reversed(range(ag_cnt)):
            it = df_stage[df_stage.optype.eq(OpType.params_all_gather) & df_stage.seq_id.eq(i)].iloc[0]
            mc, ag_op = it.mc, it.op
            if mc in mc_assigned:
                continue
            first_mb = df_comp[df_comp.mc.eq(mc) & df_comp.optype.eq(OpType.forward_compute)].start_ts.idxmin()
            deps.setdefault(df_comp.loc[first_mb].op, []).append(ag_op)
            mc_assigned.add(mc)


def setup_grads_rs_deps(deps, df_mp_group):
    """
    Make gradient reduce-scatter depend on the last bacward-compute of the same model-chunk.

    ..
        ... --> bwd-compute_{i_{vpp-1}} -->  ...   --> bwd-compute_{i_0}}
                ^                                      ^
               /                                      /  
        grads-rs_{i_{vpp-1}} --> ... --> grads-rs_{i_0}}

    , where `vpp` is the number of model-chunks in the pipeline parallelism.
    """
    pp = df_mp_group.stage.max() + 1

    for stage in range(pp):
        df_stage = df_mp_group[df_mp_group.stage.eq(stage)]
        df_comp = df_stage[df_stage.optype.isin([OpType.forward_compute, OpType.backward_compute
                                                ])].sort_values(['start_ts', 'seq_id', 'optype'])

        n_tot_rs = df_stage.optype.eq(OpType.grads_reduce_scatter).sum()

        mc_assigned = set()
        for i in range(n_tot_rs):
            it = df_stage[df_stage.optype.eq(OpType.grads_reduce_scatter) & df_stage.seq_id.eq(i)].iloc[0]
            mc, rs_op = it.mc, it.op
            if mc in mc_assigned:
                continue
            last_mb = df_comp[df_comp.mc.eq(mc) & df_comp.optype.isin([OpType.backward_compute])].end_ts.idxmax()
            deps.setdefault(rs_op, []).append(df_comp.loc[last_mb].op)
            mc_assigned.add(mc)


def get_peer_p2p_optype(optype, stage, pp):
    delta = {
        'forward-send': 1,
        'forward-recv': -1,
        'backward-send': -1,
        'backward-recv': 1,
    }
    peers = [
        ('forward-send', 'forward-recv'),
        ('backward-send', 'backward-recv'),
        ('forward-send-backward-recv', 'backward-send-forward-recv'),
        # TODO: forward-send-forward-recv
    ]
    mapper = {}
    for x, y in peers:
        mapper[x] = y
        mapper[y] = x
    for k, v in delta.items():
        if k in optype:
            return mapper[optype], (v + stage + pp) % pp
    assert False, f"get_peer_p2p_optype failed with ({optype=}, {stage=}, {pp=})"


def add_compute_comm_stream_deps(deps, send_or_recv, comm, compute):
    if send_or_recv == 'send':  # compute -> comm
        deps.setdefault(comm, []).append(compute)
    else:  # compute[i-1] -> comm -> compute[i]
        deps.setdefault(compute, []).append(comm)  # comm -> compute[i]
        last_compute_id = int(compute.split('_')[-2]) - 1
        if last_compute_id >= 0:  # compute[i-1] -> comm
            last_compute = compute.split('_')
            last_compute[-2] = str(last_compute_id)
            last_compute = '_'.join(last_compute)
            deps.setdefault(comm, []).append(last_compute)


def get_peer_op_and_match_compute(df_mp_group: pd.DataFrame, EPS=5e-4):
    """
    Pair each PP-level send/recv operation with its recv/send peers, and matches
    it with the corresponding computation operations (e.g., fwd-recv matches the fwd-compute
    that depends on the fwd-recv). 

    Args:
        df_mp_group (pd.DataFrame): A DataFrame containing ops only of one MP group 
            (i.e., dp_rank=0).
        EPS (float, optional): A small epsilon value used for timestamp comparisons 
            to account for clock synchronization errors. Defaults to 5e-4.

    Returns:
        tuple: A tuple containing:
            - cross_stream_deps (dict): A dictionary representing dependencies 
              between communication and the matched communication ops.
            - df_p2p (pd.DataFrame): A DataFrame with a p2p communication op per row
              containing: the matched compute op (`<send_or_recv>_match_compute` 
              field) and another `<send_or_recv>_p2p_group`
              field indicating the peer-to-peer group.
    """
    vpp = df_mp_group.mc.max() + 1
    pp = df_mp_group.stage.max() + 1
    df_p2p = []
    deps = {}
    for stage in range(df_mp_group.stage.max() + 1):
        df_stage = df_mp_group[df_mp_group.stage.eq(stage)]
        df_comm = df_stage[df_stage.optype.isin(OpType.p2p_optypes())].sort_values(['start_ts', 'seq_id', 'optype'])
        df_comp = df_stage[df_stage.optype.isin([OpType.forward_compute, OpType.backward_compute
                                                ])].sort_values(['start_ts', 'seq_id', 'optype'])

        df_comm = df_comm.sort_values(['start_ts', 'seq_id', 'optype'])
        # due to implementation issues, our profiler may record extra send/recv ops that weren't launched
        # we call them no-ops and need to skip them. Use `is_no_op` field, initialized to True for every op.
        for send_or_recv in ['send', 'recv']:
            df_comm[f'{send_or_recv}_is_no_op'] = True
            df_comm[f'{send_or_recv}_match_compute'] = None
        for t in ['forward', 'backward']:
            last_compute_st = df_comp[df_comp.optype == t + '-compute'].start_ts.max()
            first_compute_ed = df_comp[df_comp.optype == t + '-compute'].end_ts.min()
            for send_or_recv in ['send', 'recv']:
                compute_id = 0
                optype = t + '-' + send_or_recv
                # type 1 noop: happen in the beginning or end of the training step
                if send_or_recv == 'recv':
                    cnt = (df_comm[df_comm.optype.str.contains(optype)].start_ts <= last_compute_st + EPS).sum()
                else:
                    cnt = (df_comm[df_comm.optype.str.contains(optype)].end_ts >= first_compute_ed - EPS).sum()

                num_compute = df_comp.optype.eq(f'{t}-compute').sum()
                if cnt != num_compute:
                    assert stage in [0, pp - 1], f"Missing pp-comm ops at non-beginning or endinng {stage=}"
                    # validating for PP schedules with and without VPP
                    # - cnt=0 happens when VPP is not used since no fwd-recv/bwd-send is needed for the first stage
                    #   and likewise for the last stage
                    # - when VPP is used, usually each compute op always corrsponds to one send/recv op, even when
                    #   some are (type 2) no-ops. But sometimes 1st fwd-recv on stage 0 could be missing.
                    assert cnt == 0 or cnt + 1 == num_compute, \
                        f"Expect having only 1 fewer comm op, but actual #comm={cnt}, #compute={num_compute}"
                    if send_or_recv == 'recv' and cnt + 1 == num_compute:
                        compute_id += 1  # when the first fwd-recv no-op is missing, we simply skip it and its compute op

                for _, i in enumerate(df_comm[df_comm.optype.str.contains(optype)].index):
                    if (send_or_recv == 'recv' and df_comm.loc[i, 'start_ts'] > last_compute_st + EPS or
                            send_or_recv == 'send' and df_comm.loc[i, 'end_ts'] < first_compute_ed - EPS):
                        # type 1 no-op
                        continue
                    match_compute = f"{t}-compute_{compute_id}_{stage}"
                    gmc = df_comp[df_comp.op == match_compute]['gmc'].item()

                    # type 2 no-op are the fwd-recv (fwd-send) ops on the very first (last) model-chunk
                    # of the model, and likewise those bwd-recv and bwd-send ops. typically happens in VPP
                    _, is_no_op = get_sender_stage(t, send_or_recv, gmc, stage, vpp, pp)
                    df_comm.loc[i, f'{send_or_recv}_is_no_op'] = is_no_op
                    df_comm.loc[i, f'{send_or_recv}_match_compute'] = match_compute
                    compute_id += 1

        # pair peer ops into a p2p group
        # TODO(JK): merge send_is_no_op and recv_is_no_op into one column. Only one of them is used
        #           (send op uses send_is_no_op and likewise for recv), and the other is always intialize value (i.e., True)
        df_comm['is_no_op'] = df_comm['send_is_no_op'] & df_comm['recv_is_no_op']
        df_comm['p2p_group'] = None  # unique name for each p2p group. e.g., 'forward-recv_0_0,forward-send_0_0'
        for optype in df_comm.optype.unique():
            valid_p2p_op_id = 0
            for cnt, i in enumerate(df_comm[df_comm.optype.eq(optype)].index):
                is_no_op = df_comm.loc[i, 'is_no_op'].item()
                if not is_no_op:
                    pp_op_valid = f"{optype}_{valid_p2p_op_id}_{stage}"
                    peer_optype, peer_stage = get_peer_p2p_optype(optype, stage, pp)
                    peer_op = f"{peer_optype}_{valid_p2p_op_id}_{peer_stage}"
                    df_comm.loc[i, 'p2p_group'] = ','.join(sorted([pp_op_valid, peer_op]))
                    valid_p2p_op_id += 1

        # build compute-comm dependency
        for i in df_comm.index:
            for send_or_recv in ['send', 'recv']:
                match_compute = df_comm.loc[i, f'{send_or_recv}_match_compute']
                if match_compute is not None:  # for unidirectional comm ops like forward-send, we need to skip 'recv'
                    op = df_comm.loc[i, 'op']
                    add_compute_comm_stream_deps(deps, send_or_recv, op, match_compute)

        df_p2p.append(df_comm)
    df_p2p = pd.concat(df_p2p, ignore_index=True)
    cross_stream_deps = deps
    return cross_stream_deps, df_p2p


def setup_first_bwd_send_recv_dependency(deps: Dict, df_p2p, df: pd.DataFrame, stage):
    """
    Make the first backward send/recv depend on the second previous forward compute before the first backward compute.

    ..
        ... --> fwd-compute_{i} --> fwd-compute_{i+1} --> bwd-compute_0 --> ...
                ^
               /
        bwd-send_0, bwd-recv_0
    """
    df = df.sort_values(['start_ts', 'seq_id', 'optype'])
    for i in range(len(df)):
        if df.iloc[i]['optype'] == OpType.backward_compute:
            first_bwd_idx = i
            break
    if first_bwd_idx < 2:  # not using pp
        return
    assert (
        t :=
        df.iloc[first_bwd_idx -
                1]['optype']) == OpType.forward_compute, f"The compute op ({t}) before first backward is not forward."
    assert (t := df.iloc[first_bwd_idx - 2]['optype']
           ) == OpType.forward_compute, f"The second compute op ({t}) before first backward is not forward."
    op = df.iloc[first_bwd_idx - 2]['op']
    for t in [OpType.backward_recv, OpType.backward_send]:
        if f'{t}_0_{stage}' in df_p2p.op.unique():
            deps.setdefault(f'{t}_0_{stage}', []).append(op)


def setup_pp_dependency(df: pd.DataFrame, dsp_size: int):
    df_mp_group = df[df.dp_rank.eq(0) & df.step.eq(df.step.min())]
    dp = df.dp_rank.max() + 1
    pp = df.stage.max() + 1
    EPS = 5e-5

    compute_p2p_deps, df_p2p = get_peer_op_and_match_compute(df_mp_group, EPS)

    # 1. build cross-rank p2p communication dependency
    comm_groups = []
    for name, group in df_p2p.groupby('p2p_group'):  # no-op will have none in p2p_group hence not be iterated on
        assert len(group) == 2, f"A p2p group has more or less than 2 participants. {name=}, {group=}"
        comm_groups.append(group.op.tolist())

    ## add the rest of p2p-comm ops that are no-ops
    for pp_op in df_p2p[df_p2p.is_no_op].op.unique():
        comm_groups.append([pp_op])

    # 2. build in-rank dependency
    deps = compute_p2p_deps
    ## same stream dependency: p2p-comm
    for stage in range(df.stage.max() + 1):
        df_stage = df_mp_group[df_mp_group.stage.eq(stage)].copy()
        df_comm = df_stage[df_stage.optype.isin(OpType.p2p_optypes())].sort_values(['start_ts', 'seq_id', 'optype'])
        df_comp = df_stage[df_stage.optype.isin([OpType.forward_compute, OpType.backward_compute
                                                ])].sort_values(['start_ts', 'seq_id', 'optype'])
        setup_chain_deps(deps, df_comp.op.tolist())

        df_comm['stream'] = df_comm['optype'].map({
            'forward-send': 0,
            'forward-recv': 1,
            'backward-send': 2,
            'backward-recv': 3,
        })
        streams = [0, 1, 2, 3]
        assert not df_comm['stream'].isnull().any(), f"Unexpected optype found in {df_comm.optype.unique()}."
        for stream in streams:
            setup_chain_deps(deps, df_comm[df_comm.stream.eq(stream)].op.tolist(), last_ops=[])
        # let 1st bwd-send/recv depend on the second previous fwd-compute before the corresponding bwd-compute
        setup_first_bwd_send_recv_dependency(deps, df_p2p[df_p2p.stage.eq(stage)], df_comp, stage)

    ## same stream dependency: dp-comm
    use_dp = dp * dsp_size > 1
    if use_dp:
        for stage in range(df.stage.max() + 1):
            df_stage = df_mp_group[df_mp_group.stage.eq(stage)].copy()
            if OpType.params_all_gather in df_stage.optype:  # sometimes params_all_gather could be missing
                deps.setdefault(OpType.params_all_gather + f'_0_{stage}', [])
            df_comm = df_stage[df_stage.optype.isin([OpType.params_all_gather, OpType.grads_reduce_scatter
                                                    ])].sort_values(['start_ts', 'seq_id', 'optype'])
            setup_chain_deps(deps, df_comm.op.tolist())

    ## dependency between dp-comm & compute
    ovlp_dp = OpType.separate_grads_all_reduce in df.optype.unique()
    if use_dp:
        setup_params_ag_deps(deps, df_mp_group)
        if ovlp_dp:
            setup_grads_rs_deps(deps, df_mp_group)  # case w/o ovlp is covered in setup_post_compute_dependency
    setup_post_compute_dependency(deps, df_mp_group, ovlp_dp=ovlp_dp, use_dp=use_dp)

    # 3. build cross-rank dp communication dependency
    if use_dp:
        comm_optypes = [OpType.params_all_gather, OpType.grads_reduce_scatter, OpType.separate_grads_all_reduce]
        for op in df_mp_group[df_mp_group.optype.isin(comm_optypes)].op:
            comm_groups.append([op])

    # 4. build cross-rank embedding-grads-all-reduce dependency
    if pp > 1:
        comm_groups.append(
            [f"{OpType.embedding_grads_all_reduce}_0_0", f"{OpType.embedding_grads_all_reduce}_0_{pp - 1}"])
    else:
        comm_groups.append([f"{OpType.embedding_grads_all_reduce}_0_0"])

    # 5. build cross-rank optimizer synchronization dependency
    comm_groups.append([f"{OpType.optimizer_clip_main_grad}_0_{stage}" for stage in range(pp)])

    # build mapping from op to communication group id
    op2gid = {}
    for gid, group in enumerate(comm_groups):
        for op in group:
            op2gid[op] = gid
    return deps, comm_groups, op2gid


def label_dp_sync(df, op2gid):
    """
    Label the ops that need to be synchronized across data parallel ranks.
    """
    need_dp_sync = set()
    for _, group in df[df.optype.isin([
            OpType.params_all_gather, OpType.grads_reduce_scatter, OpType.separate_grads_all_reduce,
            OpType.optimizer_clip_main_grad
    ])].groupby('op', observed=False):
        if len(group) == 0:
            continue
        op = group.op.iloc[0]
        if op in op2gid:  # is comm:
            need_dp_sync.add(op)
    return need_dp_sync


def setup_dependency(df: pd.DataFrame, dsp_size: int):
    """
    Setup the dependency model. For efficiency purpose, only model dependencies for ops in a single step and 
    one model-parallel (MP) group (i.e., dp_rank=0), since different steps and dp ranks share the same dependency.
    This is done in `setup_pp_dependency` function. We finish the dependency model by labeling the ops
    that require DP synchronization.
    """
    p2p_ops = [i for i in df.optype.unique() if i in OpType.p2p_optypes()]
    expected_ops = set([
        OpType.forward_send,
        OpType.forward_recv,
        OpType.backward_send,
        OpType.backward_recv,
    ])
    assert set(p2p_ops).issubset(expected_ops), f"Not supported ops: {set(p2p_ops) - expected_ops}"

    dep_on_pp, comm_groups_pp, op2gid_pp = setup_pp_dependency(df, dsp_size)
    need_dp_sync = label_dp_sync(df, op2gid_pp)

    # TODO(JK): wrap below into a class
    return dep_on_pp, comm_groups_pp, op2gid_pp, need_dp_sync
