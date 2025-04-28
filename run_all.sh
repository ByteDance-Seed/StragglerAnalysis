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

rm -rf ./logs/case-study-analysis && mkdir -p ./logs/case-study-analysis # THIS IS IMPORTANT!
for trial_name in 'SE' 'ST' 'AR'; do # sequence-imbalance, stage-imbalance, artificial-slowdown
    echo ">>>>>>>>>>>running analysis for $trial_name"
    python3 ./analyzer/wia.py --metadata ./data/meta-$trial_name.yaml --trace ./data/trace-$trial_name.parquet --dump-dir ./logs/case-study-analysis --root ./logs/case-study-analysis &> ./logs/case-study-analysis/$trial_name.log
    diff ./logs/case-study-analysis/$trial_name.json ./data/result-$trial_name.json
    python3 ./analyzer/heatmap.py --trial $trial_name --dump-dir ./logs/case-study-analysis --analysis-result ./data/result-$trial_name.json >> ./logs/case-study-analysis/$trial_name.log 2>&1
    diff ./logs/case-study-analysis/heatmap-$trial_name.png ./data/heatmap-$trial_name.png
    # Ms
    python3 analyzer/compute_ms.py --trial $trial_name --dump-dir ./logs/case-study-analysis --analysis-result ./data/result-$trial_name.json --root ./logs/case-study-analysis >> ./logs/case-study-analysis/$trial_name.log 2>&1
    diff ./logs/case-study-analysis/ms-$trial_name.json ./data/ms-$trial_name.json 
    # Mw
    python3 analyzer/compute_mw.py --trial $trial_name --dump-dir ./logs/case-study-analysis --analysis-result ./data/result-$trial_name.json --root ./logs/case-study-analysis >> ./logs/case-study-analysis/$trial_name.log 2>&1
    diff ./logs/case-study-analysis/mw-$trial_name.json ./data/mw-$trial_name.json 
    # ideal timelines
    python3 analyzer/to_timeline.py ./logs/case-study-analysis/dataframe/$trial_name-0-no-blocking.parquet -o ./logs/case-study-analysis --reset-step-start --step-st 0 --step-ed 999999
    # diff "./logs/case-study-analysis/timeline-$trial_name-0-no-blocking.pkl-0-999999-[':,:,:'].json" ./data/ideal-timeline-$trial_name.json
done
echo "no diff should be printed except for png"
