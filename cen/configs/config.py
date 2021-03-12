from dataclasses import dataclass, field
import pprint as pp


class ConfigBase:

    def as_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def pretty(self):
        pp.pprint(self.as_dict())


@dataclass
class RunConfig(ConfigBase):
    """
    Miscellaneous configuration for running CEN experiments.
    Namely, the random seed to use for reproducing results.
    """
    seed: int = 1


@dataclass
class DatasetConfig(ConfigBase):
    """Arguments for loading a dataset via `cen.data.load`."""
    name: str = "mnist"
    context_kwargs: dict = field(default_factory=dict)
    feature_kwargs: dict = field(default_factory=dict)
    max_train_size: int = None
    merge_train_valid: bool = False
    permute: bool = True


@dataclass
class ModelConfig(ConfigBase):
    """Arguments for building a TensorFlow model via `cen.models.get`."""
    name: str = "baseline"
    kwargs: dict = field(default_factory=dict)


@dataclass
class NetworkConfig(ConfigBase):
    """
    Arguments for retrieving a "network" function (that takes input tensors
    and builds output tensors) via `cen.networks.get`.
    """
    kwargs: dict = field(default_factory=dict)


@dataclass
class TrainConfig(ConfigBase):
    """
    Arguments for building and training a model via
    `cen.experiment.utils.build` and `cen.experiment.train`.
    """
    loss: dict = field(default_factory=dict)
    checkpoint_kwargs: dict = field(default_factory=dict)
    tensorboard: dict = field(default_factory=dict)
    shuffle_buffer_size: int = 512
    batch_size: int = 64
    epochs: int = 1
    verbose: int = 1


@dataclass
class EvalConfig(ConfigBase):
    """
    Arguments for evaluating model performance via `cen.experiment.evaluate`.
    """
    metrics: dict = field(default_factory=dict)
    batch_size: int = 64
    verbose: int = 1


@dataclass
class CrossvalConfig(ConfigBase):
    """
    Arguments for performing cross validation via
    `cen.experiment.cross_validate`.
    """
    splits: int = 5
    shuffle: bool = False
    seed: int = None
    verbose: bool = False


@dataclass
class ExperimentConfig(ConfigBase):
    """Configuration for running CEN experiments."""

    run: RunConfig = \
        field(default_factory=RunConfig, init=True, repr=True)

    dataset: DatasetConfig = \
        field(default_factory=DatasetConfig, init=True, repr=True)

    model: ModelConfig = \
        field(default_factory=ModelConfig, init=True, repr=True)

    network: NetworkConfig = \
        field(default_factory=NetworkConfig, init=True, repr=True)

    train: TrainConfig = \
        field(default_factory=TrainConfig, init=True, repr=True)

    optimizer: dict = \
        field(default_factory=dict, init=True, repr=True)

    evaluation: EvalConfig = \
        field(default_factory=EvalConfig, init=True, repr=True)

    crossval: CrossvalConfig = \
        field(default_factory=CrossvalConfig, init=True, repr=True)

    def as_dict(self):
        return {
            "run": self.run.as_dict(),
            "dataset": self.dataset.as_dict(),
            "model": self.model.as_dict(),
            "network": self.network.as_dict(),
            "train": self.train.as_dict(),
            "optimizer": self.optimizer,
            "evaluation": self.evaluation.as_dict(),
            "crossval": self.crossval.as_dict()
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            run=RunConfig.from_dict(d["run"]),
            dataset=DatasetConfig.from_dict(d["dataset"]),
            model=ModelConfig.from_dict(d["model"]),
            network=NetworkConfig.from_dict(d["network"]),
            train=TrainConfig.from_dict(d["train"]),
            optimizer=d["optimizer"],
            evaluation=EvalConfig.from_dict(d["evaluation"]),
            crossval=CrossvalConfig.from_dict(d["crossval"])
        )

    @property
    def eval(self):
        return self.evaluation
