import random
import numpy as np

GLOBAL_SEED = 42


def set_global_seeds(seed: int = GLOBAL_SEED) -> None:
    """Set all global seeds for reproducibility.

    sklearn strategy: estimators must be instantiated with random_state=GLOBAL_SEED
    explicitly. np.random.seed covers estimators that use numpy's global state, but
    sklearn best practice is per-estimator random_state — callers are responsible for
    passing random_state=GLOBAL_SEED when constructing any sklearn estimator.
    """
    random.seed(seed)
    np.random.seed(seed)


# Apply seeds on import so any module that imports config.seeds is reproducible
set_global_seeds(GLOBAL_SEED)
