# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayLogpTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # NOTE: We copy the rollout saving from ray_ppo_trainer.py#L1282.
                    # Save before "balance_batch", since balance_batch will change the order of data
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # NOTE: kx create a reweight mask
                    reweight_mask = None

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        # NOTE: kx add the ppl metric
                        _logp_mean = (batch.batch["old_log_probs"] * response_masks).sum() / response_masks.sum()
                        ppl = torch.exp(-1.0 * _logp_mean).item()
                        metrics.update({"actor/ppl": ppl})

                        # NOTE: kx add the entropy mask
                        if self.config.algorithm.logp_diff.enable and self.config.algorithm.logp_diff.mask_by == "entropy":
                            if self.config.algorithm.logp_diff.start_step <= self.global_steps:
                                response_mask = batch.batch["response_mask"].bool()  # ensure bool type
                                print(f"****\nentropy: {entropys.shape}, {entropys.device}."
                                      f"response_mask: {response_mask.shape}, {response_mask.device}\n****")
                                # compute the entropy top-percentage mask
                                entropy_response = entropys[response_mask]  # remove padding
                                print(f"****\nentropy_response: {entropy_response.shape}, {entropy_response.device}\n****")
                                # compute the entropy mask with threshold
                                kth = int(entropy_response.shape[0] * self.config.algorithm.logp_diff.quantile)
                                kth_entropy = torch.kthvalue(entropy_response, kth).values.item()
                                entropy_mask = (entropys >= kth_entropy) & response_mask
                                metrics.update({
                                    "logp/entropy_quantile": kth_entropy,
                                    "logp/entropy_mean": entropy_response.mean().item(),
                                    "logp/entropy_mask_mean": entropy_mask.sum() / response_mask.sum(),
                                })

                                if self.config.algorithm.logp_diff.method == "clip":
                                    # we replace response mask with entropy_mask (loss mask)
                                    batch.batch["response_mask"] = entropy_mask

                        # NOTE: kx add the old_logp mask
                        if self.config.algorithm.logp_diff.enable and "old" in self.config.algorithm.logp_diff.mask_by:
                            if self.config.algorithm.logp_diff.start_step <= self.global_steps:
                                # compute the old_logp diff mask
                                _old_logp = old_log_prob.batch["old_log_probs"]
                                response_mask = batch.batch["response_mask"].bool()  # ensure bool type
                                print(f"****\nold_logp: {_old_logp.shape}, {_old_logp.device}."
                                      f"response_mask: {response_mask.shape}, {response_mask.device}\n****")
                                old_logp_response = _old_logp[response_mask]
                                print(f"****\nold_logp_response: {old_logp_response.shape}, {old_logp_response.device}\n****")
                                # compute the old_logp diff mask with threshold
                                if self.config.algorithm.logp_diff.mask_by in ["old", "old-negative"]:
                                    kth = int(old_logp_response.shape[0] * self.config.algorithm.logp_diff.quantile)
                                    kth_old_logp = torch.kthvalue(old_logp_response, kth).values.item()
                                    if self.config.algorithm.logp_diff.mask_by == "old":
                                        # valid mask is: old_logp <= kth_old_logp (lower logp, larger gradient)
                                        old_logp_mask = (_old_logp <= kth_old_logp) & response_mask
                                    else:
                                        # old-negative, valid mask is: old_logp >= kth_old_logp (higher logp, smaller gradient)
                                        old_logp_mask = (_old_logp >= kth_old_logp) & response_mask
                                    metrics.update({
                                        "logp/old_logp_quantile": kth_old_logp,
                                        "logp/old_logp_mean": old_logp_response.mean().item(),
                                        "logp/old_logp_mask_mean": old_logp_mask.sum() / response_mask.sum(),
                                    })
                                elif self.config.algorithm.logp_diff.mask_by in ['old-prob', 'old-prob-middle']:
                                    # directly use probability
                                    _old_prob = _old_logp.exp()
                                    _old_prob_response = _old_prob[response_mask]
                                    if self.config.algorithm.logp_diff.mask_by == "old-prob":
                                        # valid mask is: old_prob <= quantile
                                        old_logp_mask = (_old_prob <= self.config.algorithm.logp_diff.quantile) & response_mask
                                    elif self.config.algorithm.logp_diff.mask_by == "old-prob-middle":
                                        # valid mask is: old_prob <= upper_prob and old_prob >= lower_prob
                                        upper_prob = self.config.algorithm.logp_diff.quantile
                                        lower_prob = 1 - upper_prob
                                        old_logp_mask = (_old_prob <= upper_prob) & (_old_prob >= lower_prob) & response_mask
                                    else:
                                        raise NotImplementedError(
                                            f"Unsupported logp diff mask_by: {self.config.algorithm.logp_diff.mask_by}"
                                        )
                                    # log metrics
                                    metrics.update({
                                        "logp/old_prob_quantile": self.config.algorithm.logp_diff.quantile,
                                        "logp/old_prob_mean": _old_prob_response.mean().item(),
                                        "logp/old_prob_mask_mean": old_logp_mask.sum() / response_mask.sum(),
                                    })

                                if self.config.algorithm.logp_diff.method == "clip":
                                    # we replace response mask with old_logp_diff_mask (loss mask)
                                    batch.batch["response_mask"] = old_logp_mask
                                elif self.config.algorithm.logp_diff.method == "reweight":
                                    print("***\nUsing old_logp_mask for reweight mask, shape:", old_logp_mask.shape)
                                    reweight_mask = old_logp_mask

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                        # NOTE: kx add the logp diff mask
                        if self.config.algorithm.logp_diff.enable and "logp" in self.config.algorithm.logp_diff.mask_by:
                            if self.config.algorithm.logp_diff.start_step <= self.global_steps:
                                # compute the logp diff mask
                                _old_logp, _ref_logp = old_log_prob.batch["old_log_probs"], ref_log_prob.batch["ref_log_prob"]
                                response_mask = batch.batch["response_mask"].bool()  # ensure bool type
                                print(f"****\nold_logp: {_old_logp.shape}, {_old_logp.device}."
                                        f"ref_logp: {_ref_logp.shape}, {_ref_logp.device}."
                                        f"response_mask: {response_mask.shape}, {response_mask.device}\n****")
                                logp_diff = _old_logp - _ref_logp
                                logp_diff_response = logp_diff[response_mask]  # remove padding
                                print(f"****\nlog_diff_response: {logp_diff_response.shape}, {logp_diff_response.device}\n****")
                                # compute the logp diff mask with threshold
                                if self.config.algorithm.logp_diff.mask_by in ["logp", "logp-negative"]:
                                    # valid mask is: logp_diff >= kth_logp_diff or logp_diff <= kth_logp_diff (negative)
                                    kth = int(logp_diff_response.shape[0] * self.config.algorithm.logp_diff.quantile)
                                    kth_logp_diff = torch.kthvalue(logp_diff_response, kth).values.item()
                                    if self.config.algorithm.logp_diff.mask_by == "logp":
                                        logp_diff_mask = (logp_diff >= kth_logp_diff) & response_mask
                                    else:  # logp-negative
                                        logp_diff_mask = (logp_diff <= kth_logp_diff) & response_mask
                                
                                    metrics.update({
                                        "logp/logp_diff_quantile": kth_logp_diff,
                                        "logp/logp_diff_mean": logp_diff_response.mean().item(),
                                        "logp/logp_diff_mask_mean": logp_diff_mask.sum() / response_mask.sum(),
                                    })
                                elif self.config.algorithm.logp_diff.mask_by in ["logp-middle", "logp-end"]:
                                    # for example, quantile 0.8 will select the top 20% and bottom 20% if using logp-end
                                    # quantile 0.7 will select the middle 40% (between bottom 30% and top 30%) if using logp-middle
                                    kth_high = int(logp_diff_response.shape[0] * self.config.algorithm.logp_diff.quantile)
                                    kth_low = int(logp_diff_response.shape[0] * (1 - self.config.algorithm.logp_diff.quantile))
                                    kth_logp_diff_high = torch.kthvalue(logp_diff_response, kth_high).values.item()
                                    kth_logp_diff_low = torch.kthvalue(logp_diff_response, kth_low).values.item()
                                    if self.config.algorithm.logp_diff.mask_by == "logp-middle":
                                        # valid mask is: kth_min <= logp_diff <= kth_max
                                        logp_diff_mask = (logp_diff >= kth_logp_diff_low) & (logp_diff <= kth_logp_diff_high)
                                    else:  # logp-end, valid mask is: logp_diff <= kth_min or logp_diff >= kth_max
                                        logp_diff_mask = (logp_diff <= kth_logp_diff_low) | (logp_diff >= kth_logp_diff_high)
                                    logp_diff_mask = logp_diff_mask & response_mask  # apply response mask

                                    metrics.update({
                                        "logp/logp_diff_quantile_low": kth_logp_diff_low,
                                        "logp/logp_diff_quantile_high": kth_logp_diff_high,
                                        "logp/logp_diff_mean": logp_diff_response.mean().item(),
                                        "logp/logp_diff_mask_mean": logp_diff_mask.sum() / response_mask.sum(),
                                    })
                                else:
                                    raise NotImplementedError(
                                        f"Unsupported logp diff mask_by: {self.config.algorithm.logp_diff.mask_by}"
                                    )

                                if self.config.algorithm.logp_diff.method == "clip":
                                    # we replace response mask with logp_diff_mask (loss mask)
                                    batch.batch["response_mask"] = logp_diff_mask
                                elif self.config.algorithm.logp_diff.method == "reweight":
                                    print("***\nUsing logp_diff_mask for reweight mask, shape:", logp_diff_mask.shape)
                                    reweight_mask = logp_diff_mask


                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # NOTE: kx reweight the advantages if enabled
                    if self.config.algorithm.reweight.enable:
                        _old_logp, advantages = batch.batch["old_log_probs"], batch.batch["advantages"]
                        _old_prob = _old_logp.exp()  # convert log_prob to prob

                        _c = self.config.algorithm.reweight.constant
                        if self.config.algorithm.reweight.method == 'reshape':
                            # reshaping the weights to match the pi (1-pi) form
                            # if c == 1, weights = _old_prob
                            weights = (_c * _old_prob + 1 - _c)
                        elif self.config.algorithm.reweight.method == 'reshape_reversed':
                            # reversed reshaping, low probs will have higher weights
                            weights = (1 + _c - _c * _old_prob)
                        elif self.config.algorithm.reweight.method == 'middle':
                            # augment middle probabilities
                            weights = _c + (1 - _old_prob) * _old_prob
                        elif self.config.algorithm.reweight.method == 'normalize':
                            # default reweight method: normalize
                            # if c == 0, weights = 1 / (1 - _old_prob)
                            weights = 1 / (1 + _c - _old_prob)
                        elif self.config.algorithm.reweight.method in ['ppl', 'ppl-norm', 'ppl-norm-clamp']:
                            # reweight by ppl
                            seq_mean_logp = (_old_logp * response_masks).sum(dim=-1) / response_masks.sum(dim=-1)
                            log_seq_ppl = -1.0 * seq_mean_logp # log perplexity is -mean_logp
                            if self.config.algorithm.reweight.method in ['ppl-norm', 'ppl-norm-clamp']:
                                log_seq_ppl = log_seq_ppl.cpu().tolist()
                                # use batch.no_tensor_batch['uid'] to get each uid's ppl (so we can compute the mean & std)
                                uid2ppl = defaultdict(list)
                                assert len(batch.non_tensor_batch['uid']) == len(log_seq_ppl), \
                                    f"{len(batch.non_tensor_batch['uid'])} != {len(log_seq_ppl)}"
                                for uid, _logp_ppl in zip(batch.non_tensor_batch['uid'], log_seq_ppl, strict=True):
                                    uid2ppl[uid].append(_logp_ppl)
                                # then use mean & std to normalize
                                uid2mean, uid2std, logp_ppl_norm = {}, {}, []
                                for uid, _logp_ppl in zip(batch.non_tensor_batch['uid'], log_seq_ppl, strict=True):
                                    if uid not in uid2mean:
                                        uid2mean[uid] = np.mean(uid2ppl[uid])
                                        uid2std[uid] = np.std(uid2ppl[uid]) + 1e-6
                                    logp_ppl_norm.append((_logp_ppl - uid2mean[uid]) / uid2std[uid])
                                # convert to tensor
                                logp_ppl_norm = torch.tensor(logp_ppl_norm, device=_old_logp.device, dtype=_old_logp.dtype)
                                weights = (1 - _c * logp_ppl_norm)
                                if self.config.algorithm.reweight.method == 'ppl-norm-clamp':
                                    weights = torch.clamp(weights, min=0.8, max=1.2)
                            else:
                                weights = (1 - _c * log_seq_ppl)
                            weights = weights.unsqueeze(-1)  # expand to (bsz, 1) for later broadcasting
                            print(f"***\nUsing ppl-based reweight, shape: {weights.shape}\n***")
                        else:
                            raise NotImplementedError(
                                f"Unsupported reweight method: {self.config.algorithm.reweight.method}"
                            )

                        if self.config.algorithm.reweight.adv_mask == "positive":
                            adv_mask = advantages > 0
                        elif self.config.algorithm.reweight.adv_mask == "negative":
                            adv_mask = advantages < 0
                        else:
                            adv_mask = None  # no mask

                        if adv_mask is not None:
                            print(f"***\nUsing advantage mask, shape: {adv_mask.shape}")
                            reweight_mask = adv_mask if reweight_mask is None else reweight_mask & adv_mask

                        if reweight_mask is not None:
                            print(f"***\nUsing reweight mask, shape: {reweight_mask.shape}")
                            advantages = torch.where(reweight_mask, advantages * weights, advantages)
                        else:
                            advantages = advantages * weights
                        batch.batch["advantages"] = advantages

                        metrics.update({
                            "reweight/weights_mean": (weights * response_masks).sum() / response_masks.sum(),
                        })

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
