The branch contains **STream3R[α]**, a metric-scale model built on top of DUSt3R.  
The codebase may be somewhat unpolished.

## Checkpoint
The checkpoint can be downloaded from [here](https://huggingface.co/yslan/STream3R/tree/alpha).

## 📈 Evaluation

The evaluation follows [MonST3R](https://github.com/Junyi42/monst3r) and [Spann3R](https://github.com/HengyiWang/spann3r), [CUT3R](https://github.com/CUT3R/CUT3R).

1. Prepare Evaluation Dataset

    We follow the dataset preparation guides from [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare the datasets. For convenience, we provide the processed datasets on [Hugging Face](https://huggingface.co/datasets/yslan/pointmap_regression_evalsets), which can be downloaded directly.

    The datasets should be organized as follows under the root directiory of the project:
    ```
    data/
    ├── 7scenes
    ├── bonn
    ├── kitti
    ├── neural_rgbd
    ├── nyu-v2
    ├── scannetv2
    ├── sintel
    └── tum
    ```

2. Run Evaluation

    Use the provided scripts to evaluate different tasks.

    *Please change the model weight path accordingly.*

    ### Monodepth

    ```bash
    bash eval/monodepth/run.sh
    ```

    ### Video Depth

    ```bash
    bash eval/video_depth/run.sh
    ```

    ### Camera Pose Estimation

    ```bash
    bash eval/relpose/run.sh
    ```

    ### Multi-view Reconstruction

    ```bash
    bash eval/mv_recon/run.sh
    ```
