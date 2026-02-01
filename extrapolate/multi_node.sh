num_nodes=${1:-1}
total_size=${2:-32}
# If num_nodes is greater than 1, set seed based on the suffix of $KUBERNETES_POD_NAME
# If KUBERNETES_POD_REPLICA_TYPE == Master, then seed=0
# Otherwise, seed is the suffix id of $KUBERNETES_POD_NAME (format xxx-worker-id)
seed=0
if [ "$num_nodes" -gt 1 ]; then
  if [ "$KUBERNETES_POD_REPLICA_TYPE" == "Worker" ]; then
    seed=${KUBERNETES_POD_NAME##*-}
    seed=$((seed + 1))
  fi
fi
split_size=$((total_size / num_nodes))
echo "num_nodes: $num_nodes, total_size: $total_size, split_size: $split_size"
echo "Current seed: $seed"

shift 2
echo "Running command: --seed $seed --n $split_size $@"
python extrapolate.py --seed $seed --n $split_size $@
