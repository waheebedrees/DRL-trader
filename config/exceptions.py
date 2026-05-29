# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Exceptions
# ═══════════════════════════════════════════════════════════════════════════════

class DRLError(Exception):
    """Base exception for all DRL package errors."""


class ConfigurationError(DRLError):
    """Invalid configuration parameter."""


class InsufficientDataError(DRLError):
    """Not enough historical data for observation construction."""

    def __init__(self, required: int, available: int) -> None:
        super().__init__(
            f"Insufficient data: need {required} bars, have {available}"
        )
        self.required = required
        self.available = available


class NetworkNotInitialisedError(DRLError):
    """Network used before build() was called."""


class EnvironmentNotResetError(DRLError):
    """step() called before reset()."""


class EpisodeTerminatedError(DRLError):
    """Attempted to step a terminated episode."""

    def __init__(self, cause: str) -> None:
        super().__init__(f"Episode already terminated: {cause}")
        self.cause = cause


class TrainingError(DRLError):
    """Error during training loop."""


class CheckpointError(DRLError):
    """Model save/load failure."""


class ReplayBufferError(DRLError):
    """Replay buffer operation failure."""
