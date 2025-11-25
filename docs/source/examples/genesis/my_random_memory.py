import torch
import gymnasium
import numpy as np
from typing import Union, Tuple, Optional, List
from skrl.memories.torch import RandomMemory
from skrl.utils.spaces.torch import compute_space_size

class MyRandomMemory(RandomMemory):
    def __init__(
        self,
        memory_size,
        num_envs=1,
        device=None,
        export=False,
        export_format="pt",
        export_directory="",
        replacement=True,
        obs_space=None,
        exclude_keys=None,
        dummy_fillers=None,
    ):
        """
        Custom memory with built-in observation simplification & expansion.

        Args:
            obs_space (gymnasium.spaces.Dict): Environment observation space.
            exclude_keys (list[str]): Keys to exclude (like image/depth inputs).
            dummy_fillers (dict[str, Union[int, Tuple[int]]]): Dummy fillers for excluded keys.
        """
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory, replacement)

        # ---- CONFIG ----
        self.obs_space = obs_space
        self.exclude_keys = set(exclude_keys or {"front_depth"})
        self.dummy_fillers = dummy_fillers or {}

        # Determine simplified obs size automatically
        if obs_space is not None:
            self.simplified_obs_size = self._compute_simplified_obs_size(obs_space)
        else:
            self.simplified_obs_size = None

    # =============================================================
    # ============== HELPER METHODS ===============================
    # =============================================================

    def _compute_flat_dim(self, space):
        """Compute flattened size of a gym space (handles Box and Dict)."""
        if isinstance(space, gymnasium.spaces.Dict):
            return sum(self._compute_flat_dim(v) for v in space.spaces.values())
        elif isinstance(space, gymnasium.spaces.Box):
            return int(np.prod(space.shape))
        else:
            raise TypeError(f"Unsupported space type: {type(space)}")

    def _compute_simplified_obs_size(self, obs_space):
        """Compute total flattened size excluding excluded keys."""
        return sum(
            self._compute_flat_dim(space)
            for key, space in obs_space.spaces.items()
            if key not in self.exclude_keys
        )

    # =============================================================
    # ============== STATE EXTRACTION ==============================
    # =============================================================

    def extract_simplified_state(self, flattened_obs: torch.Tensor) -> torch.Tensor:
        """Extract simplified version of a flattened observation tensor."""
        if self.obs_space is None:
            raise ValueError("obs_space is not defined for MyRandomMemory.")

        num_envs = flattened_obs.shape[0]
        start = 0
        chunks = []

        for key, value in self.obs_space.spaces.items():
            flat_dim = int(np.prod(value.shape))
            end = start + flat_dim
            if key not in self.exclude_keys:
                chunks.append(flattened_obs[:, start:end])
            start = end

        return torch.cat(chunks, dim=-1)

    # =============================================================
    # ============== SIMPLIFIED GYM SPACE ==========================
    # =============================================================

    def simplify_gym_space(self):
        """Return a new Dict space excluding the specified keys."""
        if self.obs_space is None:
            raise ValueError("obs_space not set for MyRandomMemory.")
        return gymnasium.spaces.Dict(
            {k: v for k, v in self.obs_space.spaces.items() if k not in self.exclude_keys}
        )

    # =============================================================
    # ============== EXPAND SIMPLIFIED OBS =========================
    # =============================================================

    def expand_obs_tensor(self, small_obs: torch.Tensor) -> torch.Tensor:
        """
        Expands simplified obs tensor back into full observation tensor with dummy data.
        """
        if self.obs_space is None:
            raise ValueError("obs_space is not defined for MyRandomMemory.")

        B = small_obs.size(0)
        chunks = []
        start = 0

        for key, space in self.obs_space.spaces.items():
            flat_dim = int(np.prod(space.shape))

            if key in self.exclude_keys:
                # Fill excluded keys with dummy tensors
                filler_shape = self.dummy_fillers.get(key, flat_dim)
                chunks.append(
                    torch.zeros(B, filler_shape, dtype=small_obs.dtype, device=small_obs.device)
                )
            else:
                end = start + flat_dim
                chunks.append(small_obs[:, start:end])
                start = end

        return torch.cat(chunks, dim=-1)

    # =============================================================
    # ============== MEMORY CREATION ===============================
    # =============================================================

    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int], gymnasium.Space],
        dtype: Optional[torch.dtype] = None,
        keep_dimensions: bool = False,
    ) -> bool:
        """Create tensors for storage, simplifying obs if needed."""
        if not keep_dimensions:
            size = compute_space_size(size, occupied_size=True)

        # handle simplified obs sizes
        if name in ["states", "next_states"] and self.simplified_obs_size is not None:
            size = self.simplified_obs_size

        if name in self.tensors:
            return False

        tensor_shape = (
            (self.memory_size, self.num_envs, *size) if keep_dimensions else (self.memory_size, self.num_envs, size)
        )
        view_shape = (-1, *size) if keep_dimensions else (-1, size)

        setattr(self, f"_tensor_{name}", torch.zeros(tensor_shape, device=self.device, dtype=dtype))
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].view(*view_shape)
        self.tensors_keep_dimensions[name] = keep_dimensions

        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True

    # =============================================================
    # ============== ADDING SAMPLES ================================
    # =============================================================

    def add_samples(self, **tensors):
        """Override add_samples to automatically simplify obs."""
        if not tensors:
            raise ValueError("No samples to record.")

        tmp = tensors.get("states", tensors[next(iter(tensors))])
        dim, shape = tmp.ndim, tmp.shape

        # multi-environment samples
        if dim > 1 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    if name in ["states", "next_states"]:
                        tensor = self.extract_simplified_state(tensor)
                    self.tensors[name][self.memory_index].copy_(tensor)
            self.memory_index += 1

        # single env / others handled as before
        elif dim > 1 and self.num_envs == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    num_samples = min(shape[0], self.memory_size - self.memory_index)
                    self.tensors[name][self.memory_index : self.memory_index + num_samples].copy_(
                        tensor[:num_samples].unsqueeze(1)
                    )
                    self.memory_index = (self.memory_index + num_samples) % self.memory_size

        else:
            raise ValueError(f"Unexpected shape {shape}")

        # reset indices and handle wrap-around
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True
            if self.export:
                self.save(directory=self.export_directory, format=self.export_format)

    # =============================================================
    # ============== SAMPLING =====================================
    # =============================================================

    def sample_all(self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1):
        """Sample all data and expand obs back if needed."""
        if sequence_length > 1:
            raise NotImplementedError("Sequences not supported in this example.")

        total = self.memory_size * self.num_envs
        if mini_batches > 1:
            batch_size = total // mini_batches
            batches = [(i * batch_size, (i + 1) * batch_size) for i in range(mini_batches)]
        else:
            batches = [(0, total)]

        results = []
        for start, end in batches:
            mini = []
            for name in names:
                data = self.tensors_view[name][start:end]
                if name in ["states", "next_states"]:
                    data = self.expand_obs_tensor(data)
                mini.append(data)
            results.append(mini)
        return results
