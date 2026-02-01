import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" # make the scheduling deterministic
import time
import json
import sys
# sys.path.append('..')
import argparse
import datasets
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from copy import deepcopy
from math_utils import get_acc_list
from typing import List, Tuple
from scipy.special import softmax, logsumexp


def weighted_sampling(outputs, outputs_assistant, enable_mask:List[bool], sampling_params:SamplingParams, weights:Tuple[float, float]=(1.05, -0.05)):
    """
    Perform weighted sampling based on the logprobs from two models.
    """
    sampled_next_tokens, resample_infos = [], []
    for output, output_assistant, enable in zip(outputs, outputs_assistant, enable_mask):
        if not enable:
            # If the weighted sampling is disabled, we just use the main model's output
            sampled_next_tokens.append(output.outputs[0].token_ids[0])
            resample_infos.append(None)
        else:
            # 0-idx means the first (and only) response and token respectively
            out, out_assistant = output.outputs[0], output_assistant.outputs[0]
            lp_info, lp_info_assistant = out.logprobs[0], out_assistant.logprobs[0]
            token_ids, logprobs = [key for key in lp_info], np.array([lp.logprob for lp in lp_info.values()])
            logp_map_assistant = {key: lp.logprob for key, lp in lp_info_assistant.items()}
            # 1. align the assistant's logprobs with the main model's token_ids
            pad_lp = min(logp_map_assistant.values())
            aligned_logprobs_assistant = np.array([logp_map_assistant[tid] if tid in logp_map_assistant \
                                                else pad_lp for tid in token_ids])
            # 2. reweight the logprobs & apply temperature
            logits = logprobs * weights[0] + aligned_logprobs_assistant * weights[1]
            logits /= sampling_params.temperature 
            # 3. top-p sampling
            logits_sorted_indices = np.argsort(logits)
            logits_sorted = logits[logits_sorted_indices]
            top_p_mask = ~(np.cumsum(softmax(logits_sorted)) <= 1 - sampling_params.top_p)
            top_p_mask[-1] = True  # Ensure the last token (highest prob) is always included
            sampled_idx = np.random.choice(logits_sorted_indices[top_p_mask], p=softmax(logits_sorted[top_p_mask]))
            sampled_next_tokens.append(token_ids[sampled_idx])
            resample_infos.append({
                'original': out.token_ids[0],
                'assistant': out_assistant.token_ids[0],
                'resampled': token_ids[sampled_idx]
            })
    return sampled_next_tokens, resample_infos

def get_enable_mask(outputs, outputs_assistant, criteria:str="none", threshold:float=1.0):
    """
    Get a mask indicating whether to use the assistant's logprobs based on the criteria.
    """
    enable_mask = []
    for output, output_assistant in zip(outputs, outputs_assistant):
        out, out_assistant = output.outputs[0], output_assistant.outputs[0]
        lp_info, lp_info_assistant = out.logprobs[0], out_assistant.logprobs[0]
        if criteria == "entropy":
            # Calculate entropy for the main model's logprobs
            logprobs = np.array([lp.logprob for lp in lp_info.values()])
            logps_norm = logprobs - logsumexp(logprobs)
            entropy = -np.sum(np.exp(logps_norm) * logps_norm)
            enable_mask.append(entropy >= threshold)
        elif criteria == "assist_entropy":
            logprobs = np.array([lp.logprob for lp in lp_info_assistant.values()])
            logps_norm = logprobs - logsumexp(logprobs)
            entropy = -np.sum(np.exp(logps_norm) * logps_norm)
            enable_mask.append(entropy >= threshold)
        elif criteria in ["logp", "neg_logp"]:
            # Calculate logp diff between the main model and the assistant
            next_token = out.token_ids[0]
            logp_main = lp_info[next_token].logprob
            logp_assistant = lp_info_assistant[next_token].logprob if next_token in lp_info_assistant \
                             else min([lp.logprob for lp in lp_info_assistant.values()])
            logp_diff = logp_main - logp_assistant
            enabled = logp_diff <= threshold if criteria == "logp" else -logp_diff <= threshold
            enable_mask.append(enabled)
        elif criteria in ['kl', 'kl_r' ,'js']:
            # KL, Reversed KL, and JS divergence
            def kl_divergence(lp_info, lp_info_assistant):
                token_ids, logprobs = [key for key in lp_info], np.array([lp.logprob for lp in lp_info.values()])
                logp_map_assistant = {key: lp.logprob for key, lp in lp_info_assistant.items()}
                pad_lp = min(logp_map_assistant.values())
                aligned_logprobs_assistant = np.array([logp_map_assistant[tid] if tid in logp_map_assistant \
                                                       else pad_lp for tid in token_ids])
                logprobs_norm = logprobs - logsumexp(logprobs)
                aligned_logprobs_assistant_norm = aligned_logprobs_assistant - logsumexp(aligned_logprobs_assistant)
                # KL divergence
                kl = np.sum(np.exp(logprobs_norm) * (logprobs_norm - aligned_logprobs_assistant_norm))
                return kl
            kl_main = kl_divergence(lp_info, lp_info_assistant)
            kl_assistant = kl_divergence(lp_info_assistant, lp_info)
            avg_divergence = 0.5 * (kl_main + kl_assistant)
            if criteria == 'kl':
                enable_mask.append(kl_assistant >= threshold)
            elif criteria == 'kl_r':
                enable_mask.append(kl_main >= threshold)
            elif criteria == 'js':  # js-style average divergence
                enable_mask.append(avg_divergence >= threshold)
        elif criteria == "none":
            enable_mask.append(False)
        elif criteria == "all":
            enable_mask.append(True)
        elif criteria == "rand":
            enable_mask.append(np.random.rand() < threshold)
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
    return enable_mask

def decode_with_two_models(args:argparse.Namespace, llm:LLM, llm_assistant:LLM, prompts:List[str], gts:List[str]):
    tokenizer = llm.get_tokenizer()
    initial_prompts = sum([[tokenizer.encode(p, add_special_tokens=False) for p in prompts]] * args.n, [])
    all_generated_texts = {i:[] for i in range(len(initial_prompts))}
    all_resampled_masks = {i:[] for i in range(len(initial_prompts))}
    all_resample_infos = {i:{} for i in range(len(initial_prompts))} # idx -> info dict
    print(f"Total prompts: {len(initial_prompts)}\nExample:\n{tokenizer.decode(initial_prompts[0])}")

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=1,
        logprobs=args.logprobs,
        seed=args.seed,
        skip_special_tokens=False
    )
    window_size = len(prompts)
    # window_size = len(initial_prompts)
    cur_prompts = deepcopy(initial_prompts[:window_size])
    cur_idxs = list(range(len(cur_prompts)))
    cur_lengths = [0] * len(cur_prompts)
    max_idx = max(cur_idxs) + 1
    pbar = tqdm(total=len(initial_prompts), desc="Processing prompts")
    while cur_idxs:
        # We generate one token for each prompt in cur_prompts
        # If any response reaches EOS or the max tokens, we remove it from cur_idxs & cur_prompts, and add new prompts & idx into cur_prompts & cur_idxs
        start = time.time()
        inputs = [TokensPrompt(prompt_token_ids=p) for p in cur_prompts]
        outputs = llm.generate(inputs, sampling_params=sp, use_tqdm=False)
        outputs_assistant = llm_assistant.generate(inputs, sampling_params=sp, use_tqdm=False)
        # reweighting
        # We use some criteria to decide whether to use the assistant's logprobs
        # 1. entropy: if the entropy is higher than a threshold (e.g., 0.8)
        # 2. logp diff: if the logp diff is lower than a threshold (e.g., -0.2)
        # 3. neg_logp diff: if the negative logp diff is higher than a threshold (e.g., 0.2)
        # 4. none/all: no or all enabled
        enable_mask = get_enable_mask(outputs, outputs_assistant, criteria=args.criteria, threshold=args.threshold)
        sampled_next_tokens, resample_infos = weighted_sampling(outputs, outputs_assistant, enable_mask, sp, weights=args.weights)
        # update responses, prompts, and lengths
        finished_idxs = []
        for i, next_token in enumerate(sampled_next_tokens):
            all_generated_texts[cur_idxs[i]].append(next_token) # append next token
            all_resampled_masks[cur_idxs[i]].append(enable_mask[i]) # append whether the assistant's logprobs were used
            if resample_infos[i]: all_resample_infos[cur_idxs[i]][cur_lengths[i]] = resample_infos[i]  # store resample info
            cur_prompts[i] += [next_token]
            cur_lengths[i] += 1
            if next_token == tokenizer.eos_token_id or cur_lengths[i] >= args.max_tokens:
                finished_idxs.append(i)
        for i in reversed(finished_idxs):  # reverse to avoid index issues
            cur_idxs.pop(i)
            cur_prompts.pop(i)
            cur_lengths.pop(i)
        pbar.update(len(finished_idxs))
        # Then we add new prompts if there are still prompts to generate
        while max_idx < len(initial_prompts) and len(cur_prompts) < window_size:
            cur_prompts.append(deepcopy(initial_prompts[max_idx]))
            cur_idxs.append(max_idx)
            cur_lengths.append(0)
            max_idx += 1
        pbar.set_postfix_str(f"len: {np.mean(cur_lengths):.1f} | {len(cur_prompts)/(time.time()-start):.1f} tok/s (resampled {np.mean(enable_mask):.0%})")
    pbar.close()

    all_info, avg_accs, avg_resampled = [], [], []
    for idx, gt in enumerate(gts):
        # we get all the generated texts for the each prompt (which is repeated n times)
        cur_generated_texts = [all_generated_texts[i*len(prompts)+idx] for i in range(args.n)]
        cur_generated_texts = [tokenizer.decode(t, skip_special_tokens=False) for t in cur_generated_texts]
        cur_resample_rates = [np.mean(all_resampled_masks[i*len(prompts)+idx]) for i in range(args.n)]
        cur_text_lengths = [len(all_resampled_masks[i*len(prompts)+idx]) for i in range(args.n)] # the length is equal to mask size
        cur_resample_infos = [all_resample_infos[i*len(prompts)+idx] for i in range(args.n)]
        # cur_weighted_generated_idxs = [np.where(np.array(t) == True)[0].tolist() for t in cur_weighted_generated]
        accs = get_acc_list(cur_generated_texts, gt)
        all_info.append({
            'prompt': prompts[idx],
            'gt': gt,
            'generated_texts': cur_generated_texts,
            'accs': accs,
            'lengths': cur_text_lengths,
            'resample_rates': cur_resample_rates,
            'resample_infos': cur_resample_infos,
        })
        avg_accs.append(np.mean(accs))
        avg_resampled.append(np.mean(cur_resample_rates))
    print(f"Average accuracy: {np.mean(avg_accs):.4f} ± {np.std(avg_accs):.4f}")
    print(f"Average resampling rate: {np.mean(avg_resampled):.4f} ± {np.std(avg_resampled):.4f}")
    return all_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode with two models")
    parser.add_argument("--model", type=str, default="../hf_models/DAPO-Qwen-32B", help="Path to the first model")
    parser.add_argument("--assistant", type=str, default="../hf_models/Qwen2.5-32B", help="Path to the second model")
    parser.add_argument("--dataset", type=str, default="../hf_datasets/aime24", help="Path to the dataset")
    parser.add_argument("--tokenizer", type=str, default="model", choices=["model", "assistant"],
                        help="Which tokenizer to use for encoding the prompts")
    # sampling params
    parser.add_argument("--n", type=int, default=32, help="#responses per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p for sampling")
    parser.add_argument("--logprobs", type=int, default=50, help="Number of logprobs to return")
    parser.add_argument("--max_tokens", type=int, default=20_000, help="Maximum tokens to generate")
    parser.add_argument("--gpu_util", type=float, default=0.4, help="GPU utilization for the model")
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    # weights for the two models
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 0.0), 
                        help="Weights for the two models' logprobs, e.g., 1.05 -0.05")
    # whether to enable the assistant's logprobs
    parser.add_argument("--criteria", type=str, default="none", choices=["entropy", "assist_entropy", "logp", "neg_logp", "none", "all", "rand", "js", "kl", "kl_r"])
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold for the criteria")
    # other args
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()
    np.random.seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if sum(args.weights) != 1.0:
        print(f"Warning: weights {args.weights} do not sum to 1.0, normalizing them.")
        args.weights = [w / sum(args.weights) for w in args.weights]
    save_path = f"{os.path.basename(args.model)}-{os.path.basename(args.assistant)}-{os.path.basename(args.dataset)}"
    save_path += f"-w{args.weights[0]}_{args.weights[1]}-select_{args.criteria}-thresh_{args.threshold}"
    save_path += f"-n{args.n}-t{args.temperature}-top_p{args.top_p}"
    os.makedirs(os.path.join(args.save_dir, save_path), exist_ok=True)
    save_path = os.path.join(save_path, f"seed{args.seed}.json")
    print(f"Results will be saved to {os.path.join(args.save_dir, save_path)}")
    
    # Load dataset
    if 'aime24' in args.dataset:
        ds = datasets.load_dataset(args.dataset, split='train')
    elif 'aime25' in args.dataset:
        ds = datasets.load_dataset(args.dataset, split='test')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    prompts, gts = ds['problem'], ds['answer']

    # Load models
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True, 
              enforce_eager=True, tensor_parallel_size=args.tp_size, max_logprobs=args.logprobs)
    assistant = LLM(model=args.assistant, gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True, 
              enforce_eager=True, tensor_parallel_size=args.tp_size, max_logprobs=args.logprobs)

    # process the prompts
    tokenizer = llm.get_tokenizer() if args.tokenizer == "model" else assistant.get_tokenizer()
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q.strip() + suffix}], tokenize=False, 
                                             add_generation_prompt=True) for q in prompts]
    
    info = decode_with_two_models(args, llm, assistant, prompts, gts)
    json.dump(info, open(os.path.join(args.save_dir, save_path), "w"), indent=4, ensure_ascii=False)
