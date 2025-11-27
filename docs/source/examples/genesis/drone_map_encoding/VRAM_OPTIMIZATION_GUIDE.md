# VRAM Optimization Guide for Parallel Training

## Current Memory Analysis (1 env on RTX 3050 Laptop - 4GB VRAM)

### Memory Breakdown Per Environment:
1. **Lidar sensor data (during rollout)**: 64×64×3×4 bytes = 49 KB per env
2. **Model weights**: ~2-5 MB (shared across all envs)
3. **Optimizer states**: 2× model weights = ~4-10 MB
4. **Memory buffer (1000 steps)**:
   - Without lidar (your current setup): 1000 × 17 × 4 bytes = 68 KB per env ✅
   - With lidar (if stored): 1000 × 12305 × 4 bytes = 49 MB per env ❌
5. **Forward pass activations**: ~10-20 MB per env (depends on batch size)
6. **Genesis scene/physics**: Variable, likely 100-500 MB base + per env overhead

## Issues Found in Your Code

### ❌ Issue 1: Attention Weights Still Computed
**Location:** `drone_train.py:201-206`

```python
attn_out, attn_weights = self.attn(...)
del attn_weights  # Too late - already allocated!
```

**Memory Cost:** 5 envs × 4096 positions × 4 heads × 4 bytes = 320 KB per forward pass

**Fix:**
```python
attn_out, _ = self.attn(
    query=proprio_encoded.unsqueeze(1),
    key=map_encoded,
    value=map_encoded,
    need_weights=False  # ✅ Don't allocate at all
)
attn_out = attn_out.squeeze(1)
```

---

### ❌ Issue 2: Unnecessary Intermediate Tensors in CNN
**Location:** `drone_train.py:145-159`

The CNN creates intermediate feature maps that consume memory.

**Current Memory:** 5 envs × 64×64 × (16+32+32+1) channels × 4 bytes = 6.5 MB

**Fix:** Use in-place operations and gradient checkpointing:

```python
def forward_cnn(self, map_scans):
    """Memory-efficient CNN forward with gradient checkpointing."""
    from torch.utils.checkpoint import checkpoint
    
    # Use checkpoint to trade compute for memory during backward pass
    return checkpoint(self._forward_cnn_impl, map_scans, use_reentrant=False)

def _forward_cnn_impl(self, map_scans):
    B = map_scans.shape[0]
    x = map_scans.permute(0, 3, 1, 2)
    
    cnn_feats = self.cnn(x)
    height = self.height_layer(x)
    
    combined = torch.cat([cnn_feats, height], dim=1)
    flat_feats = combined.view(B, combined.shape[1], -1).permute(0, 2, 1)
    return flat_feats
```

---

### ❌ Issue 3: Shared Model Cache Holds Gradients
**Location:** `drone_train.py:217, 221`

```python
self._shared_output = shared_out.detach()  # Good, but can be better
```

**Fix:** Use `torch.no_grad()` context:

```python
def compute(self, inputs, role):
    # ... existing code ...
    
    if role == "policy":
        with torch.no_grad():
            self._shared_output = shared_out.clone()  # Explicit copy without grad
        return self.mean_layer(shared_out), self.log_std_parameter, {}
    
    elif role == "value":
        if self._shared_output is not None:
            value_input = self._shared_output
            self._shared_output = None
        else:
            value_input = shared_out
        return self.value_layer(value_input), {}
```

---

### ❌ Issue 4: Large Rollout Buffer
**Location:** `drone_train.py:250`

```python
cfg["rollouts"] = 8  # memory_size
```

With 1000 memory size, you're storing 1000 steps per env.

**Fix:** Reduce rollout size:

```python
cfg["rollouts"] = 8  # Keep this
memory = MyRandomMemory(
    memory_size=16,  # Reduce from 1000 to 16 (2× rollouts)
    num_envs=env.num_envs,
    obs_space=env.observation_space,
    exclude_keys=["front_depth"],
    dummy_fillers={"front_depth": 64*64*3},
)
```

**Memory Saved:** (1000 - 16) × 17 × 4 bytes × N_envs = ~67 KB per env

---

### ❌ Issue 5: Lidar Resolution Too High
**Location:** `drone_env.py:107-108`

```python
self.lidar = self.scene.add_sensor(gs.sensors.DepthCamera(
    pattern=gs.sensors.DepthCameraPattern(res=(64,64)),  # 4096 points
    **sensor_kwargs
))
```

**Memory Cost:** 64×64×3 = 12,288 values per env during rollout

**Fix:** Reduce resolution:

```python
self.lidar = self.scene.add_sensor(gs.sensors.DepthCamera(
    pattern=gs.sensors.DepthCameraPattern(res=(32,32)),  # 1024 points (75% reduction)
    **sensor_kwargs
))
```

**Update model and config:**
```python
# In drone_train.py
map_shape=(32, 32, 3)  # Line 101
dummy_fillers={"front_depth": 32*32*3}  # Line 236
```

---

### ❌ Issue 6: Mini-batches Too Small
**Location:** `drone_train.py:252`

```python
cfg["mini_batches"] = 4
```

Smaller mini-batches = more gradient accumulation steps = more memory overhead.

**Fix:** Increase mini-batches (if memory allows):

```python
cfg["mini_batches"] = 8  # or even 16
```

---

### ✅ Issue 7: State Preprocessor Allocations
**Location:** `drone_train.py:269-272`

The `RunningStandardScaler` allocates buffers for mean/std tracking.

**Optimization:** Use shared preprocessor or disable if not critical:

```python
# Option 1: Disable preprocessor (if you're already normalizing in env)
cfg["state_preprocessor"] = None
cfg["value_preprocessor"] = None

# Option 2: Keep but ensure it's efficient
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space, 
    "device": device,
    "epsilon": 1e-8,  # Default is fine
}
```

---

## Recommended Configuration for Multi-Env Training

### For 5 Parallel Environments on 4GB VRAM:

```python
# drone_train.py modifications

# 1. Reduce lidar resolution
# In drone_env.py line 107:
self.lidar = self.scene.add_sensor(gs.sensors.DepthCamera(
    pattern=gs.sensors.DepthCameraPattern(res=(32,32)),
    **sensor_kwargs
))

# 2. Update model config
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2, reduction="sum",
                 map_shape=(32, 32, 3),  # ✅ Changed from (64,64,3)
                 proprio_dim=13,
                 map_feat_dim=16,  # ✅ Reduced from 32
                 attn_heads=2):  # ✅ Reduced from 4
        # ... rest of init ...

# 3. Smaller CNN
self.cnn = nn.Sequential(
    nn.Conv2d(self.C, 8, kernel_size=3, padding=1), nn.ELU(),  # 16→8
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ELU(),      # 32→16
    nn.Conv2d(16, map_feat_dim, kernel_size=3, padding=1), nn.ELU()
)

# 4. Smaller MLP
self.mlp = nn.Sequential(
    nn.Linear(self.flat_mlp_input_dim, 128), nn.ELU(),  # 256→128
    nn.Linear(128, 64), nn.ELU(),                        # 128→64
    nn.Linear(64, 32), nn.ELU()                          # 64→32, new layer
)
self.mean_layer = nn.Linear(32, self.num_actions)  # 64→32
self.value_layer = nn.Linear(32, 1)                # 64→32

# 5. Reduce memory size
memory = MyRandomMemory(
    memory_size=16,  # ✅ Reduced from 1000
    num_envs=env.num_envs,
    obs_space=env.observation_space,
    exclude_keys=["front_depth"],
    dummy_fillers={"front_depth": 32*32*3},  # ✅ Updated
)

# 6. Optimize PPO config
cfg["rollouts"] = 8
cfg["learning_epochs"] = 3  # ✅ Reduced from 5
cfg["mini_batches"] = 8     # ✅ Increased from 4
cfg["grad_norm_clip"] = 0.5  # ✅ Reduced from 1.0

# 7. Disable preprocessors if possible
cfg["state_preprocessor"] = None
cfg["value_preprocessor"] = None

# 8. Use gradient checkpointing (add to Shared model)
def forward_cnn(self, map_scans):
    from torch.utils.checkpoint import checkpoint
    return checkpoint(self._forward_cnn_impl, map_scans, use_reentrant=False)

def _forward_cnn_impl(self, map_scans):
    # Move existing forward_cnn code here
    B = map_scans.shape[0]
    x = map_scans.permute(0, 3, 1, 2)
    cnn_feats = self.cnn(x)
    height = self.height_layer(x)
    combined = torch.cat([cnn_feats, height], dim=1)
    flat_feats = combined.view(B, combined.shape[1], -1).permute(0, 2, 1)
    return flat_feats

# 9. Fix attention weights
attn_out, _ = self.attn(
    query=proprio_encoded.unsqueeze(1),
    key=map_encoded,
    value=map_encoded,
    need_weights=False  # ✅ Critical!
)
```

---

## Expected Memory Savings

| Optimization | Memory Saved (per env) | Total for 5 envs |
|--------------|------------------------|------------------|
| Lidar 64→32 | 37 KB | 185 KB |
| Memory 1000→16 | 67 KB | 335 KB |
| CNN channels reduced | 3.2 MB | 16 MB |
| MLP size reduced | 0.5 MB | 2.5 MB |
| Attention heads 4→2 | 160 KB | 800 KB |
| Gradient checkpointing | 2-5 MB | 10-25 MB |
| **TOTAL** | **~6-9 MB** | **~30-45 MB** |

---

## Additional Tips

### 1. Monitor VRAM Usage
```python
import torch

def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Add to training loop
if timestep % 100 == 0:
    print_memory_stats()
```

### 2. Clear Cache Periodically
```python
# In training loop
if timestep % 500 == 0:
    torch.cuda.empty_cache()
```

### 3. Use Mixed Precision (if supported)
```python
# Check if SKRL supports AMP
cfg["use_amp"] = True  # Automatic Mixed Precision
```

### 4. Reduce Genesis Scene Complexity
```python
# In drone_env.py
self.scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),  # Reduce from 2
    # ... other options ...
)
```

### 5. Profile Memory Usage
```python
# Add to drone_train.py
torch.cuda.memory._record_memory_history(enabled=True)

# After training
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

---

## Testing Strategy

Start with fewer envs and gradually increase:

```bash
# Test with increasing env counts
python drone_train.py --num_envs=1  # Baseline
python drone_train.py --num_envs=2
python drone_train.py --num_envs=3
python drone_train.py --num_envs=5
python drone_train.py --num_envs=8  # If VRAM allows
```

Monitor VRAM at each step to find the sweet spot.
