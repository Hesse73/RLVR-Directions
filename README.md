# RLVR Directions
> Source code for our ICLR'26 paper: [On the Direction of RLVR Updates for LLM Reasoning: Identification and Exploitation](https://openreview.net/forum?id=r6Pw3RiMYL)

## Extrapolate

The extrapolate code is in the `extrapolate` folder.
We generate 32 responses for each prompt, with 8 different random seeds and 4 responses per seed, the script can be found in `scripts/run_extrapolate.sh`.

The results will be saved in the `extrapolate/results` folder, which can be checked using the `check_results.ipynb` notebook.


## RL Training w/ Reweighting

We use the verl library to perform RL training with reweighting.
We modify the DAPO recipe of verl, and the modified code is in the `verl/recipe/logp_rl` folder.

The training script for Qwen2.5-Math-7B and Qwen3-8B-Base are in the `scripts/run_7B.sh` and `scripts/run_8B.sh` respectively.

Please follow [the DAPO's recipe](https://github.com/BytedTsinghua-SIA/DAPO) to prepare the data and models, then you can run the scripts to start training.

## Reference
Please cite our work if you find it helpful!
```Bibtex
@inproceedings{
  huang2026on,
  title={On the Direction of {RLVR} Updates for {LLM} Reasoning: Identification and Exploitation},
  author={Kexin Huang and Haoming Meng and Junkang Wu and Jinda Lu and Chiyu Ma and Ziqian Chen and Xue Wang and Bolin Ding and Jiancan Wu and Xiang Wang and Xiangnan He and Guoyin Wang and Jingren Zhou},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=r6Pw3RiMYL}
}
```
