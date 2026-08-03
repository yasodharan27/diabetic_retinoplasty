"""
Generic optimizer / learning-rate-schedule construction, reusable across
modules. No model or dataset assumptions -- these just wrap
`tf.keras.optimizers` and `tf.keras.optimizers.schedules`.
"""

import tensorflow as tf

_OPTIMIZER_REGISTRY = {
    "adam": tf.keras.optimizers.Adam,
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
}


def build_optimizer(name="adam", learning_rate=1e-4, **kwargs):
    """Factory: resolve an optimizer by name for config-driven training scripts."""
    key = name.lower()
    if key not in _OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {sorted(_OPTIMIZER_REGISTRY)}")
    return _OPTIMIZER_REGISTRY[key](learning_rate=learning_rate, **kwargs)


def cosine_decay_schedule(initial_learning_rate, decay_steps, alpha=0.0):
    """Proactive cosine-decay LR schedule, usable as an alternative to the
    reactive ReduceLROnPlateau callback when the total step count is known
    ahead of time."""
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate, decay_steps=decay_steps, alpha=alpha,
    )


def exponential_decay_schedule(initial_learning_rate, decay_steps, decay_rate=0.96, staircase=False):
    """Proactive exponential-decay LR schedule."""
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate, decay_steps=decay_steps,
        decay_rate=decay_rate, staircase=staircase,
    )


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay -- common for transformer-style
    models (relevant to the project's Swin Transformer components)."""

    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        self._cosine = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate, decay_steps=decay_steps, alpha=alpha,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_lr = self.initial_learning_rate * (step / tf.maximum(warmup_steps, 1.0))
        return tf.cond(
            step < warmup_steps,
            lambda: warmup_lr,
            lambda: self._cosine(step - warmup_steps),
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
        }


def warmup_cosine_schedule(initial_learning_rate, warmup_steps, decay_steps, alpha=0.0):
    return WarmupCosineSchedule(initial_learning_rate, warmup_steps, decay_steps, alpha=alpha)
