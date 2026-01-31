# Reproducibility Guide

This document explains the reproducibility guarantees of this codebase and how to achieve deterministic results.

## What is Guaranteed

### Procedural Reproducibility

Given:

- Same code (git commit)
- Same CLI arguments
- Same base seed (`--seed`)

You will get:

- **Identical self-play game sequences**: Same actions taken in each game
- **Identical evaluation match schedules**: Same matchups, same game order
- **Identical training data**: Same observations, policies, and value targets
- **Identical SGD batch ordering**: Same shuffling of training examples

### Determinism Scope

| Component                  | Reproducible? | Notes                              |
| -------------------------- | ------------- | ---------------------------------- |
| Self-play action sequences | ✅ Yes        | With same seed + config            |
| Eval match outcomes        | ✅ Yes        | With same seed + config            |
| Training data generation   | ✅ Yes        | With same seed + config            |
| SGD batch ordering         | ✅ Yes        | Uses `np.random.default_rng(seed)` |
| Agent MCTS decisions       | ✅ Yes        | Uses agent-local `random.Random`   |
| Belief sampler             | ✅ Yes        | Uses per-sampler `random.Random`   |

## What is NOT Guaranteed

### Bitwise Identical Weights

Even with `--deterministic-torch`, you may NOT get bitwise-identical model weights across:

- Different GPU models (e.g., RTX 3090 vs A100)
- Different CUDA/cuDNN versions
- Different floating-point precision modes
- CPU vs GPU training

This is due to:

- Non-deterministic GPU atomics in some CUDA operations
- Different algorithm selections by cuDNN autotuning
- Floating-point associativity differences

### Cross-Platform Reproducibility

Results may differ between:

- Linux vs Windows vs macOS
- Different Python versions
- Different PyTorch versions

## How to Run in Deterministic Mode

### Training

```bash
python -m scripts.train.main \
    --seed 42 \
    --deterministic-torch \
    --device cpu \
    --games 10 \
    --T 8 \
    --S 4
```

The `--deterministic-torch` flag enables:

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)` (if available)

**Warning**: Deterministic mode may significantly reduce GPU performance.

### Evaluation

```bash
python -m scripts.eval.match \
    --seed 42 \
    --model models/model.pt \
    --a azbsmcts \
    --b bsmcts \
    --n 20
```

Evaluation is deterministic by default when using the same seed.

## Seed Derivation

All seeds are derived deterministically from a single base seed using `derive_seed()`:

```python
from scripts.common import seeding

# For training self-play, game index 5, player 0
seed = seeding.derive_seed(
    base_seed=42,
    purpose="train/agent",
    game_idx=5,
    player_id=0,
)
```

### Seed Namespaces

| Purpose          | Description                            |
| ---------------- | -------------------------------------- |
| `train/selfplay` | Self-play game generation (deprecated) |
| `train/agent`    | Per-agent RNG in training              |
| `train/belief`   | Belief sampler in training             |
| `train/sgd`      | SGD batch shuffling                    |
| `eval/agent`     | Per-agent RNG in evaluation            |
| `eval/belief`    | Belief sampler in evaluation           |
| `eval/rng`       | Random baseline in evaluation          |
| `api/agent`      | Per-agent RNG in API                   |
| `api/belief`     | Belief sampler in API                  |
| `api/rng`        | API backend random fallback            |

## Configuration Logging

Every run automatically logs:

1. **config.json**: Full configuration snapshot

   ```json
   {
     "config": {
       "seed": 42,
       "device": "cuda:0",
       "deterministic_torch": true,
       ...
     },
     "fingerprint": {
       "python_version": "3.11.5",
       "platform": "Linux-5.15.0-x86_64",
       "torch_version": "2.1.0",
       "cuda_version": "12.1",
       "cudnn_version": "8902",
       "device_name": "NVIDIA GeForce RTX 3090",
       "git_commit": "abc123def456",
       "git_dirty": false
     }
   }
   ```

2. **Console output**: Reproducibility fingerprint printed at startup
   ```
   [seeding] base_seed=42 deterministic_torch=ON
   === Reproducibility Fingerprint ===
     Python: 3.11.5
     Platform: Linux-5.15.0-x86_64
     PyTorch: 2.1.0
     CUDA: 12.1
     cuDNN: 8902
     Device: NVIDIA GeForce RTX 3090
     Git: abc123def (dirty)
   ===================================
   ```

## Verifying Reproducibility

### Unit Tests

Run the reproducibility tests:

```bash
pytest tests/test_reproducibility.py -v
```

### Manual Verification

1. Run training twice with the same seed:

   ```bash
   python -m scripts.train.main --seed 42 --games 5 --logdir run1
   python -m scripts.train.main --seed 42 --games 5 --logdir run2
   ```

2. Compare the training metrics:

   ```bash
   diff run1/train_metrics.jsonl run2/train_metrics.jsonl
   ```

   The files should be identical (on the same machine/environment).

## Best Practices

1. **Always specify `--seed`**: Don't rely on default seeds for experiments.

2. **Log your exact command**: Include in experiment notes or use a shell script.

3. **Save the config.json**: It contains all hyperparameters and the reproducibility fingerprint.

4. **Use version control**: The git commit hash of your _source code_ is logged automatically, so you can trace which code version produced which results. Training outputs (runs/, models/) should remain in `.gitignore`.

5. **Document your environment**: The fingerprint helps, but also consider using Docker or conda environments.

6. **For publications**: Include seed, git commit, and environment details in supplementary materials.

## Troubleshooting

### "Results differ between runs"

1. Check that you're using the same `--seed`
2. Verify git commit matches (`git rev-parse HEAD`)
3. Compare fingerprints in config.json files
4. Ensure no global RNG usage (search for `np.random.` without `default_rng`)

### "GPU results differ from CPU"

This is expected. Use `--device cpu` for exact reproducibility, or accept minor differences on GPU.

### "Results differ after code change"

Any code change affecting RNG consumption order will change results. This is by design - same code + same seed = same results.
