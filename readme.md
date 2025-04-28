<div align="center">
 👋 Hi, everyone!
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="豆包研究员 - 小红书">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

# Artifact for Paper "Understanding Stragglers in Large Model Training Using What-if Analysis"
<p align="center">
  <a href="XXX">
    <img src="https://img.shields.io/badge/License-Apache2.0-blue"></a>
</p>

## Introduction
This artifact provides the core functionality of the simulator and the what-if analysis proposed in the paper, along with three sample traces to demonstrate the usage of the tool. The expected output includes the following for each sample trace:
- Estimated slowdown $S$ (i.e., Eq. (1))
- Slowdown $S_t$ attributed to each operation type $t$ (i.e., Eq. (2))
- Slowdown $S_w$ attributed to each worker $w$ (i.e., Eq. (4))
- Characterization metrics $M_W$(i.e., Eq. (5)) and $M_S$ for individual worker issues and stage partitioning imbalance, respectively
- A heatmap visualization as in Fig. 14.
- A timeline of the simulated ideal trace visualizable in Perfetto.

## Code Structure
```bash
├── analyzer  # Analyzer codes
├── data # Stores input data for analysis and corresponding expected results
├── logs
│   └── case-study-analysis # Reproduced results should be generated here and match those in the ../data directory
├── format.sh # Script for code formatting
├── readme.md
├── requirements.txt # Python dependencies
├── style.yapf # Configuration file for code formatting, defining the code style
└── run_all.sh # Convenient script used to reproduce all the results for each trace
```

`./analyzer/wia.py` is the entry-point for the what-if analysis. It takes in one job's trace and outputs various slowdown estimatation through simulation. We document in detail what each output field mean in the `AnalyzerResult` class in `./analyzer/metatypes.py`.

In `data` folder there are three traces named by root causes `SE`, `ST` and `AR`, denoting sequence length imbalance, stage partitioning imbalance, and artifial individual worker slowdown (see section A.2), respectively. They correspond to the jobs analyzed in Sections 5.2, 5.3 and A.2 (the one with highest level of slowdown), respectively.
Several files are included for each trace:
- `meta-<trace_name>.yaml`: metadata for the corresonding job.
- `trace-<trace_name>.parquet`: trace data. Each row corresponds to a recorded operation with the following fields:
  * `dp_rank`: DP rank of the worker performing the op
  * `stage`: PP rank of the worker
  * `rank`: global rank of the worker
  * `step`: training step this op is at
  * `optype`: operation type. See `metatypes.py` for all supported types.
  * `start_ts`: start timestamp
  * `duration`: duration of the op
  * `seq_id`: ops of the same type in a step on a worker will be assigned a sequence number `seq_id` in ascending order of their start times
  * `mc`: model chunk (virtual stage) ID, the ID within the PP stage
  * `gmc`: global model chunk (virtual stage) ID, the ID in the model. e.g., PP_size=VPP_size=2, then PP0 holds model chunks with `gmc=0` and `2`, while PP1 holds `gmc=1` and `3`.
  * `mb_id`: microbatch ID, only valid for forward-compute and backward-compute ops
- `result-<trace_name>.json`: expected what-if analysis result.
- `heatmap-<trace_name>.png`: expected heatmap generated with the analysis result.
- `ms-<trace_name>.json`: the $M_S$ metric as in Section 5.2 of the paper.
- `mw-<trace_name>.json`: the $M_W$ metric as in Eq. (5) of the paper.
- `timeline-<trace_name>.json.gz`: the timeline of the original trace that can be visualized in Perfetto.
- `ideal-timeline-<trace_name>.json.gz`: the expected timeline of the simulated ideal that can be visualized in Perfetto. One could contrast it with the original timeline to have a more intuitive understanding on the simulation.

## Evaluation Steps
### 1.Install dependencies.
  The code is tested with Python 3.11. To install the necessary dependencies, run the following command:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Execute the reproduction script.

  For each trace we will analyze with `wia.py`, produce heatmap with `heatmap.py`, compute the $M_S$ and $M_W$ metrics with `compute_ms.py` and `compute_mw.py`, and generate the ideal timeline with `to_timeline.py`. For convienience we pack them all in one script and all you need is run it:
  ```bash
  ./run_all.sh
  ```

### 3. Check if output is expected.

  The script above also compares with the expected results in `data` folder. The result should only differ (if any) in the generated heatmap PNG files, and only in figure plotting but not the underlying data, potentially due to plotting logic difference across platforms/versions.

  Result highlight:
  - For trace AR, $M_W$ should be large (~100%) as it's individual worker issues, and only one worker should be highlighted in the heatmap.
  - For trace ST, $M_S$ should be large (~120%) as it's caused by long last stage, and only workers on the last stage will be highlighted in the heatmap.
  - For trace SE, $M_S$ is interestingly high (~68%) as well, since it also suffers from the issue of long last stage, second to the dominant sequence length imbalance. In the heatmap all workers are hightlighted as this is a randomly occuring issue.

### 4. Optional: explore customized what-if analysis
  Users can also easily run their own analysis using our tool. We show two examples in `custom-wia.ipynb`. Simply run the file to see the result, or play with it with your own analysis.

## License
This project is licensed under Apache 2.0. See the LICENSE flie for details.


## About [ByteDance Seed Team](https://team.doubao.com/)
Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.