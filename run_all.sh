#!/bin/bash

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

[ -d "./logs/case-study-analysis" ] && rm -rf ./logs/case-study-analysis
mkdir -p ./logs/case-study-analysis
for trace_name in 'SE' 'ST' 'AR'; do # sequence-imbalance, stage-imbalance, artificial-slowdown
    out=./logs/case-study-analysis/$trace_name.log
    echo ">>>>>>>>>>>running analysis for trace $trace_name. check $out for output logs"
    echo ">> running wia.py"
    python ./analyzer/wia.py --metadata ./data/meta-$trace_name.yaml --trace ./data/trace-$trace_name.parquet --dump-dir ./logs/case-study-analysis --root ./logs/case-study-analysis &> $out
    diff ./logs/case-study-analysis/$trace_name.json ./data/result-$trace_name.json
    # heatmap
    echo ">> running heatmap"
    python ./analyzer/heatmap.py --trial $trace_name --dump-dir ./logs/case-study-analysis --analysis-result ./data/result-$trace_name.json >> $out 2>&1
    diff ./logs/case-study-analysis/heatmap-$trace_name.png ./data/heatmap-$trace_name.png
    # Ms
    echo ">> computing Ms"
    python analyzer/compute_ms.py --trial $trace_name --dump-dir ./logs/case-study-analysis --analysis-result ./data/result-$trace_name.json --root ./logs/case-study-analysis >> $out 2>&1
    diff ./logs/case-study-analysis/ms-$trace_name.json ./data/ms-$trace_name.json 
    # Mw
    echo ">> computing Mw"
    python analyzer/compute_mw.py --trial $trace_name --dump-dir ./logs/case-study-analysis --analysis-result ./data/result-$trace_name.json --root ./logs/case-study-analysis >> $out 2>&1
    diff ./logs/case-study-analysis/mw-$trace_name.json ./data/mw-$trace_name.json 
    # ideal timelines
    echo ">> dumping timeline"
    python analyzer/to_timeline.py ./logs/case-study-analysis/dataframe/$trace_name-0-no-blocking.parquet -o ./logs/case-study-analysis --reset-step-start --step-st 0 --step-ed 999999 >> $out 2>&1
    # diff "./logs/case-study-analysis/timeline-$trace_name-0-no-blocking.pkl-0-999999-[':,:,:'].json" ./data/ideal-timeline-$trace_name.json
done
echo "no diff should be printed except for png"
