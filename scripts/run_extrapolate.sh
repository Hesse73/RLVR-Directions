# We use 8 different random seeds to generate 4 responses for each seed (total 32 responses)

# DAPO: threshold -0.3, weight 0.05
for (( seed=0; seed<=7; seed++ )) do
    python extrapolate.py --seed $seed --n 4 --model "../hf_models/Qwen2.5-32B" --assistant "../hf_models/DAPO-Qwen-32B" --tokenizer "assistant" --criteria neg_logp --threshold -0.3 --weights -0.05 1.05 --save_dir "results" # extrapolation
    python extrapolate.py --seed $seed --n 4 --model "../hf_models/Qwen2.5-32B" --assistant "../hf_models/DAPO-Qwen-32B" --tokenizer "assistant" --criteria neg_logp --threshold -0.3 --weights 0.0 1.0 --save_dir "results" # replace only (no extrapolate)
done

# ORZ: threshold -0.4, weight 0.1
for (( seed=0; seed<=7; seed++ )) do
    python extrapolate.py --seed $seed --n 4 --model "../hf_models/Qwen2.5-32B" --assistant "../hf_models/Open-Reasoner-Zero-32B" --tokenizer "assistant" --criteria neg_logp --threshold -0.4 --weights -0.1 1.1 --save_dir "results" # extrapolation
    python extrapolate.py --seed $seed --n 4 --model "../hf_models/Qwen2.5-32B" --assistant "../hf_models/Open-Reasoner-Zero-32B" --tokenizer "assistant" --criteria neg_logp --threshold -0.4 --weights 0.0 1.0 --save_dir "results" # replace only (no extrapolate)
done

# UniReason: threshold -0.35, weight 0.1
for (( seed=0; seed<=7; seed++ )) do
    python extrapolate.py --seed $seed --n 4 --model "../hf_models/Qwen3-14B-Base" --assistant "../hf_models/UniReason-Qwen3-14B-RL" --tokenizer "assistant" --criteria neg_logp --threshold -0.35 --weights -0.1 1.1 --save_dir "results" # extrapolation
    python extrapolate.py --seed $seed --n 4 --model "../hf_models/Qwen3-14B-Base" --assistant "../hf_models/UniReason-Qwen3-14B-RL" --tokenizer "assistant" --criteria neg_logp --threshold -0.35 --weights 0.0 1.0 --save_dir "results" # replace only (no extrapolate)
done

