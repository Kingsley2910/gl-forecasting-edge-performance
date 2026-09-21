import math
import numpy as np
from enum import IntEnum
from pathlib import Path
from typing import Optional
from collections import deque

import json
import random

from gossiplearning.config import TrainingConfig, HistoryConfig
from gossiplearning.log import Logger
from gossiplearning.models import (
    StopCriterion,
    MergeStrategy,
    ModelWeights,
    Loss,
    NodeId,
    MetricValue,
    MetricName,
    ModelBuilder,
    NodeDataFn,
    Dataset,
    Link,
    AggregatorFn,
    WeightsMessage,
    LabelledData,
    NodeWeightFn,
)
from gossiplearning.weights_marshaling import (
    MarshalWeightsFn,
    unflatten_weights,
)
from utils.metrics import compute_metrics, Metrics
from utils.janossy import prepare_janossy_input, prepare_janossy_test_input, UncertaintyTrackingCallback


class NodeState(IntEnum):
    ACTIVE = 0
    STOPPED = 1
    TRAINING = 2


class Node:
    """
    A gossip learning node.
    """

    def __init__(
        self,
        *,
        create_model_fn: ModelBuilder,
        id: NodeId,
        links: tuple[Link, ...],
        training_config: TrainingConfig,
        history_config: HistoryConfig,
        workspace_dir: Path,
        logger: Logger,
        node_data_fn: NodeDataFn,
        aggregator: AggregatorFn,
        marshal_weights_fn: MarshalWeightsFn,
        test_set: LabelledData,
        weight_fn: NodeWeightFn,
    ) -> None:
        """
        Initialize the node for gossip protocol.

        :param create_model_fn: the function used for creating a model
        :param id: the node identifier
        :param links: the set of node links
        :param training_config: the global training & gossip configuration
        :param workspace_dir: the workspace base directory
        """

        self.id = id
        # internal state

        self._model = create_model_fn()
        self._create_model = create_model_fn
        self._logger = logger

        self._training_config = training_config
        self._history_config = history_config
        self._workspace_dir = workspace_dir

        self.data: Dataset = node_data_fn(id)
        self.node_type = self._extract_node_type(self.data["X_train"])
        # Dataset ready for training: one entry per remote node type.
        self._synthetic_data_by_node_type: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] = {}

        # Prediction blocks, ordered from oldest to newest.
        # Inputs are identical across blocks of the same type.
        self._synthetic_history_by_node_type: dict[
            int, tuple[np.ndarray, deque]
        ] = {}

        if self._training_config.synthetic_data_source is not None:
            self.load_synthetic_data(Path(self._training_config.synthetic_data_source))

        self._last_improved_time = 0
        self._updates_without_improving = 0
        self._best_val_loss: Loss = math.inf
        self._best_weights: Optional[ModelWeights] = None
        self._completed_updates = 0

        self._received_weights: dict[NodeId, WeightsMessage] = {}
        self._aggregator = aggregator
        self._marshal_weights_fn = marshal_weights_fn
        self._test_set = test_set

        # public state
        self.accumulated_weight = 0
        self.active_links = list(links)
        self.training_history: dict[MetricName, list[MetricValue]] = {}
        self.state = NodeState.ACTIVE
        self.n_training_samples = len(self.data["X_train"])
        self.eval_metrics: list[Metrics] = []
        self.weight = weight_fn(self.data)

        self._tracker_dir = self._workspace_dir / "plots" / "uncertainty" / f"node_{self.id}"
        self._tracker_dir.mkdir(parents=True, exist_ok=True)

        self._tracker = UncertaintyTrackingCallback(
            plot_interval=1,
            save_path=str(self._tracker_dir))
        
    def _extract_node_type(self, X: np.ndarray) -> int:
        """
        Extract the node type from the node's local training data.

        Each valid sequence step has the structure:
        [rate, function one-hot encoding, node_type].

        Padding steps contain only zeros.
        """
        if X.ndim != 3:
            raise ValueError(
                f"Expected X_train with 3 dimensions, received shape {X.shape}"
            )

        # A valid step has at least one non-zero value before node_type.
        # Padding steps are entirely zero.
        valid_steps = np.any(X[:, :, :-1] != 0, axis=2)

        node_types = X[:, :, -1][valid_steps]

        if len(node_types) == 0:
            raise ValueError(
                f"Cannot determine node_type for node {self.id}: "
                "no valid steps found in X_train."
            )

        unique_node_types = np.unique(node_types)

        if len(unique_node_types) != 1:
            raise ValueError(
                f"Node {self.id} contains multiple node types: "
                f"{unique_node_types.tolist()}"
            )

        return int(unique_node_types[0])
    
    def _replace_node_type(
        self,
        X: np.ndarray,
        new_node_type: int,
    ) -> np.ndarray:
        """
        Create a copy of X and replace node_type in every real sequence step.

        Padding steps remain filled with zeros.
        """
        if new_node_type not in (0, 1, 2, 3, 4, 5):
            raise ValueError(
                f"Invalid node_type {new_node_type}. Expected a value from 0 to 5."
            )

        X_modified = X.copy()

        # True for actual function steps, False for padding.
        valid_steps = np.any(X_modified[:, :, :-1] != 0, axis=2)

        # Replace node_type only in real steps.
        X_modified[:, :, -1] = np.where(
            valid_steps,
            new_node_type,
            0,
        )

        return X_modified
    
    def _build_received_model(
        self,
        message: WeightsMessage,
    ):
        """
        Create a temporary model containing the weights received from another node.

        The local model is not modified.
        """
        received_model = self._create_model()

        received_weights = unflatten_weights(
            received_model,
            message.marshaled_weights.weights,
        )

        received_model.set_weights(received_weights)

        return received_model
    
    def _generate_synthetic_data(
        self,
        message: WeightsMessage,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data using the received model.

        The local workloads are preserved, while node_type is replaced
        with the node_type of the sender.
        """
        received_node_type = message.sender_node_type

        if received_node_type not in (0, 1, 2, 3, 4, 5):
            raise ValueError(
                f"Invalid sender node_type: {received_node_type}"
            )

        # Same local workloads, but with the sender's node_type.
        X_synthetic = self._replace_node_type(
            self.data["X_train"],
            new_node_type=received_node_type,
        )

        # Reconstruct the received model.
        received_model = self._build_received_model(message)

        # Prepare input for Janossy pooling.
        X_prepared = prepare_janossy_test_input(
            X_synthetic,
            num_permutations=6,
        )

        # The received model predicts:
        # - 3 regression targets
        # - overloaded_node probability
        predictions = received_model.predict(
            X_prepared,
            verbose=0,
        )

        regression_predictions = np.asarray(predictions[0])

        # Same classification rule used in centralized evaluation:
        # probability < 0.5 -> 0
        # probability >= 0.5 -> 1
        classification_predictions = np.array(
            [
                0 if float(np.asarray(p).squeeze()) < 0.5 else 1
                for p in predictions[1]
            ],
            dtype=int,
        ).reshape(-1, 1)

        if regression_predictions.ndim != 2:
            raise RuntimeError(
                "Regression predictions must be a 2-dimensional array. "
                f"Received shape: {regression_predictions.shape}"
            )

        if regression_predictions.shape[1] != 3:
            raise RuntimeError(
                "Expected 3 regression targets "
                "(cpu_usage_node, ram_usage_node, "
                "ram_usage_node_percentage), "
                f"received {regression_predictions.shape[1]}."
            )

        # Y order:
        # 0 -> cpu_usage_node
        # 1 -> ram_usage_node
        # 2 -> ram_usage_node_percentage
        # 3 -> overloaded_node
        Y_synthetic = np.concatenate(
            [
                regression_predictions,
                classification_predictions,
            ],
            axis=1,
        )

        if len(X_synthetic) != len(Y_synthetic):
            raise RuntimeError(
                "Synthetic X and Y have different numbers of samples: "
                f"{len(X_synthetic)} != {len(Y_synthetic)}"
            )

        if Y_synthetic.shape[1] != 4:
            raise RuntimeError(
                "Synthetic Y must contain 4 columns, "
                f"received shape {Y_synthetic.shape}."
            )

        return X_synthetic, Y_synthetic

    def _store_synthetic_block(
        self,
        node_type: int,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        """Append a block; automatically discard the oldest when full."""
        if node_type not in (0, 1, 2, 3, 4, 5):
            raise ValueError(f"Invalid synthetic node type: {node_type}")

        if node_type == self.node_type:
            raise ValueError("Synthetic blocks must represent a remote type.")

        expected_X = self._replace_node_type(
            self.data["X_train"],
            new_node_type=node_type,
        )

        # Row-wise aggregation requires exactly matching inputs and order.
        if X.shape != expected_X.shape or not np.array_equal(X, expected_X):
            raise ValueError(
                f"Synthetic inputs for type {node_type} do not match "
                "the current local workloads and sample order."
            )

        if Y.shape != (len(X), 4):
            raise ValueError(
                f"Expected synthetic Y shape {(len(X), 4)}, got {Y.shape}."
            )

        if not np.isfinite(Y).all():
            raise ValueError("Synthetic targets contain NaN or infinite values.")

        if not np.isin(Y[:, -1], [0, 1]).all():
            raise ValueError("Synthetic classification labels must be 0 or 1.")

        if node_type not in self._synthetic_history_by_node_type:
            self._synthetic_history_by_node_type[node_type] = (
                X.copy(),
                deque(maxlen=self._training_config.synthetic_window_size),
            )

        _, blocks = self._synthetic_history_by_node_type[node_type]
        blocks.append(Y.copy())
        self._refresh_synthetic_dataset(node_type)

    def _refresh_synthetic_dataset(self, node_type: int) -> None:
        """Build the training dataset from the retained prediction blocks."""
        X, blocks = self._synthetic_history_by_node_type[node_type]

        # Shape: (number_of_blocks, number_of_samples, 4).
        predictions = np.stack(list(blocks), axis=0)

        if self._training_config.synthetic_dataset_mode == "concatenate":
            X_ready = np.concatenate([X] * len(blocks), axis=0)
            Y_ready = np.concatenate(list(blocks), axis=0)

        else:
            # Average each regression target independently.
            regression = predictions[:, :, :3].mean(axis=0)

            labels = predictions[:, :, 3]
            positive_votes = labels.sum(axis=0)
            block_count = len(blocks)

            classification = (2 * positive_votes > block_count).astype(int)

            # If votes tie, use the newest block's class.
            ties = 2 * positive_votes == block_count
            classification[ties] = labels[-1, ties].astype(int)

            X_ready = X
            Y_ready = np.column_stack((regression, classification))

        self._synthetic_data_by_node_type[node_type] = (X_ready, Y_ready)


    def persist_synthetic_data(self) -> None:
        """Save both the training dataset and its rolling history."""
        synthetic_dir = (
            self._workspace_dir
            / "synthetic_data"
            / f"node_{self.id}"
        )
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        for node_type, (X_base, blocks) in (
            self._synthetic_history_by_node_type.items()
        ):
            X_ready, Y_ready = self._synthetic_data_by_node_type[node_type]

            np.savez_compressed(
                synthetic_dir / f"node_type_{node_type}.npz",
                format_version=np.array(2),
                # Keep X/Y available for tools reading the training dataset.
                X=X_ready,
                Y=Y_ready,
                # Preserve individual blocks to resume the rolling window.
                X_base=X_base,
                Y_blocks=np.stack(list(blocks), axis=0),
            )

        print(
            f"[SYNTHETIC_SAVE] Node {self.id}: "
            f"saved types={list(self._synthetic_history_by_node_type)} "
            f"in {synthetic_dir}"
        )


    def load_synthetic_data(self, source_dir: Path) -> None:
        """Load rolling histories, or a single block from legacy files."""
        node_dir = source_dir / f"node_{self.id}"
        if not node_dir.is_dir():
            raise FileNotFoundError(
                f"Synthetic data directory not found: {node_dir}"
            )

        files = sorted(node_dir.glob("node_type_*.npz"))
        if not files:
            raise RuntimeError(
                f"No synthetic data files found in {node_dir}"
            )

        for data_file in files:
            node_type = int(data_file.stem.removeprefix("node_type_"))

            with np.load(data_file, allow_pickle=False) as data:
                if "format_version" in data:
                    if int(data["format_version"].item()) != 2:
                        raise ValueError(
                            f"Unsupported synthetic format in {data_file}"
                        )
                    X = data["X_base"]
                    Y_blocks = data["Y_blocks"]
                else:
                    # Previous format: one synthetic block per type.
                    X = data["X"]
                    Y_blocks = data["Y"][None, ...]

            if (
                Y_blocks.ndim != 3
                or Y_blocks.shape[0] == 0
                or Y_blocks.shape[1:] != (len(X), 4)
            ):
                raise ValueError(
                    f"Invalid synthetic history shape in {data_file}: "
                    f"{Y_blocks.shape}"
                )

            # Replace any existing history for this type.
            self._synthetic_history_by_node_type.pop(node_type, None)
            self._synthetic_data_by_node_type.pop(node_type, None)

            # Keep the newest K blocks if loading a larger window.
            window_size = self._training_config.synthetic_window_size
            for Y in Y_blocks[-window_size:]:
                self._store_synthetic_block(node_type, X, Y)

        print(
            f"[SYNTHETIC_LOAD] Node {self.id}: "
            f"loaded types={list(self._synthetic_history_by_node_type)} "
            f"from {node_dir}"
        )

    def _diagnostics_enabled(self):
        return (
            self._training_config.merge_strategy
            == MergeStrategy.NODE_TYPE_MERGE
        )

    def _diagnostics_dir(self):
        folder = self._workspace_dir / "diagnostics" / f"node_{self.id}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _diagnostics_evaluate(self, model, stage):
        """Evaluate on fixed local inputs, without changing training data."""
        if not self._diagnostics_enabled():
            return

        # Prepare fixed permutations only once.
        # Restore Python/NumPy RNG states afterwards.
        if not hasattr(self, "_diagnostics_inputs"):
            python_state = random.getstate()
            numpy_state = np.random.get_state()

            try:
                random.seed(12345)
                np.random.seed(12345)

                self._diagnostics_inputs = {
                    split: prepare_janossy_test_input(
                        self.data[f"X_{split}"],
                        num_permutations=6,
                    )
                    for split in ("train", "val")
                }
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

        folder = self._diagnostics_dir()
        update = self._completed_updates + 1

        record = {
            "update": update,
            "plot_index": update - 1,
            "stage": stage,
        }
        arrays = {}

        for split, inputs in self._diagnostics_inputs.items():
            truth = np.asarray(self.data[f"Y_{split}"])

            regression_batches = []
            probability_batches = []

            # Direct inference, with dropout disabled.
            # Batch to avoid evaluating the entire dataset at once.
            batch_size = self._training_config.batch_size

            for start in range(0, len(inputs), batch_size):
                predictions = model(
                    inputs[start:start + batch_size],
                    training=False,
                )
                regression_batches.append(np.asarray(predictions[0]))
                probability_batches.append(np.asarray(predictions[1]))

            regression = np.concatenate(regression_batches, axis=0)
            probabilities = np.concatenate(
                probability_batches, axis=0
            ).reshape(-1).astype(np.float64)

            labels = truth[:, -1].astype(int)
            predicted_labels = (probabilities > 0.5).astype(int)

            if not np.isfinite(probabilities).all():
                raise ValueError("Non-finite diagnostic probabilities")

            # Diagnostic BCE computed directly from probabilities.
            p = np.clip(probabilities, 1e-7, 1 - 1e-7)
            sample_bce = -(
                labels * np.log(p)
                + (1 - labels) * np.log1p(-p)
            )

            valid_rows = ~np.all(
                np.isnan(truth[:, :3]), axis=1
            )

            mse = None
            if valid_rows.any():
                mse = float(np.mean(
                    (
                        truth[valid_rows, :3]
                        - regression[valid_rows]
                    ) ** 2
                ))

            record[split] = {
                "samples": len(labels),
                "accuracy": float(np.mean(predicted_labels == labels)),
                "bce": float(np.mean(sample_bce)),
                "mse": mse,
                "true_negative": int(np.sum(
                    (labels == 0) & (predicted_labels == 0)
                )),
                "false_positive": int(np.sum(
                    (labels == 0) & (predicted_labels == 1)
                )),
                "false_negative": int(np.sum(
                    (labels == 1) & (predicted_labels == 0)
                )),
                "true_positive": int(np.sum(
                    (labels == 1) & (predicted_labels == 1)
                )),
            }

            # Row order matches the original local dataset.
            arrays[f"{split}_truth"] = truth
            arrays[f"{split}_probabilities"] = probabilities
            arrays[f"{split}_sample_bce"] = sample_bce
            arrays[f"{split}_regression"] = regression

        prefix = folder / f"update_{update:03d}_{stage}"

        prefix.with_suffix(".json").write_text(
            json.dumps(record, indent=2)
        )
        np.savez_compressed(
            prefix.with_suffix(".npz"),
            **arrays,
        )

    def _diagnostics_synthetic_changes(self, previous, senders):
        """Compare the old and new synthetic datasets used for training."""
        if not self._diagnostics_enabled():
            return

        changes = []

        for node_type, (_, new_y) in (
            self._synthetic_data_by_node_type.items()
        ):
            old_y = previous.get(node_type)

            item = {
                "node_type": int(node_type),
                "new_type": old_y is None,
                "samples_after": len(new_y),
                "overloaded_count_after": int(
                    np.sum(new_y[:, -1] == 1)
                ),
                "overloaded_fraction_after": float(
                    np.mean(new_y[:, -1])
                ),
                "retained_blocks": len(
                    self._synthetic_history_by_node_type[node_type][1]
                ),
            }

            if old_y is not None:
                item["samples_before"] = len(old_y)
                item["overloaded_fraction_before"] = float(
                    np.mean(old_y[:, -1])
                )

                # Row-by-row comparison is valid for aggregate:
                # same local workloads, same order, same remote type.
                if (
                    self._training_config.synthetic_dataset_mode
                    == "aggregate"
                    and old_y.shape == new_y.shape
                ):
                    old_labels = old_y[:, -1]
                    new_labels = new_y[:, -1]

                    item["changed_labels"] = int(np.sum(
                        old_labels != new_labels
                    ))
                    item["changed_fraction"] = float(np.mean(
                        old_labels != new_labels
                    ))
                    item["zero_to_one"] = int(np.sum(
                        (old_labels == 0) & (new_labels == 1)
                    ))
                    item["one_to_zero"] = int(np.sum(
                        (old_labels == 1) & (new_labels == 0)
                    ))
                    item["regression_mean_absolute_change"] = (
                        np.mean(
                            np.abs(new_y[:, :3] - old_y[:, :3]),
                            axis=0,
                        ).tolist()
                    )

            changes.append(item)

        update = self._completed_updates + 1

        record = {
            "update": update,
            "plot_index": update - 1,
            "senders": senders,
            "model_update_mode":
                self._training_config.node_type_model_update,
            "synthetic_mode":
                self._training_config.synthetic_dataset_mode,
            "local_samples": len(self.data["X_train"]),
            "local_overloaded_count": int(
                np.sum(self.data["Y_train"][:, -1] == 1)
            ),
            "synthetic_samples": sum(
                len(y)
                for _, y in self._synthetic_data_by_node_type.values()
            ),
            "changes": changes,
        }

        path = (
            self._diagnostics_dir()
            / f"update_{update:03d}_synthetic_changes.json"
        )
        path.write_text(json.dumps(record, indent=2))

    def merge_models(self) -> None:
        """
        Merge all the received model weights into the current model.

        The internal model weights are updated. The number of trained samples is set at the
        maximum between the number of trained samples of the merged models.

        With NODE_TYPE_MERGE, each received model is first used to generate
        synthetic data representing its node type. Synthetic data for the
        same node type replace the previously stored block.
        """
        diagnostics_previous = {}
        diagnostics_senders = []

        if self._diagnostics_enabled():
            self._diagnostics_pending = True

            diagnostics_previous = {
                node_type: y.copy()
                for node_type, (_, y)
                in self._synthetic_data_by_node_type.items()
            }

            diagnostics_senders = [
                {
                    "node": int(sender),
                    "node_type": int(message.sender_node_type),
                }
                for sender, message in self._received_weights.items()
            ]

            self._diagnostics_evaluate(
                self._model,
                "before_merge",
            )

        messages = tuple(self._received_weights.values())

        if self._training_config.merge_strategy == MergeStrategy.NODE_TYPE_MERGE:
            for message in messages:
                received_node_type = message.sender_node_type

                # Do not generate synthetic data for the local node type,
                # because real local data are already available.
                if received_node_type == self.node_type:
                    continue

                X_synthetic, Y_synthetic = self._generate_synthetic_data(
                    message
                )

                self._store_synthetic_block(
                    received_node_type,
                    X_synthetic,
                    Y_synthetic,
                )

                _, blocks = self._synthetic_history_by_node_type[
                    received_node_type
                ]
                X_ready, _ = self._synthetic_data_by_node_type[
                    received_node_type
                ]

                print(
                    f"[NODE_TYPE_MERGE] Node {self.id} "
                    f"received type={received_node_type}; "
                    f"blocks={len(blocks)}/"
                    f"{self._training_config.synthetic_window_size}; "
                    f"mode={self._training_config.synthetic_dataset_mode}; "
                    f"training_samples_for_type={len(X_ready)}"
                )
                # -----------------------------

        messages = tuple(self._received_weights.values())

        if (
            self._training_config.merge_strategy
            == MergeStrategy.NODE_TYPE_MERGE
        ):
            mode = self._training_config.node_type_model_update
            alpha = self._training_config.node_type_alpha

            if mode == "keep_local":
                # Keep local parameters and accumulated weight.
                pass

            elif mode == "overwrite" or (
                mode == "interpolate" and alpha == 1.0
            ):
                # Preserve the existing overwrite behavior exactly.
                self._model, self.accumulated_weight = self._aggregator(
                    self._model,
                    self.accumulated_weight,
                    messages,
                )

            elif mode == "interpolate":
                if len(messages) != 1:
                    raise RuntimeError(
                        "Interpolation requires exactly one received model."
                    )

                # alpha=0 preserves the existing keep_local behavior.
                if alpha > 0.0:
                    local_weights = self._model.get_weights()

                    received_weights = unflatten_weights(
                        self._model,
                        messages[0].marshaled_weights.weights,
                    )

                    interpolated_weights = [
                        (
                            local + alpha * (received - local)
                        ).astype(local.dtype, copy=False)
                        for local, received in zip(
                            local_weights,
                            received_weights,
                        )
                    ]

                    self._model.set_weights(interpolated_weights)

                    # Bookkeeping convention for a mixed model:
                    # retain the larger accumulated weight.
                    self.accumulated_weight = max(
                        self.accumulated_weight,
                        messages[0].model_weight,
                    )

            else:
                raise ValueError(
                    f"Unknown node_type_model_update: {mode}"
                )

        else:
            # Other merge strategies keep their existing behavior.
            self._model, self.accumulated_weight = self._aggregator(
                self._model,
                self.accumulated_weight,
                messages,
            )

        if self._diagnostics_enabled():
            self._diagnostics_synthetic_changes(
                diagnostics_previous,
                diagnostics_senders,
            )
            self._diagnostics_evaluate(
                self._model,
                "after_merge",
            )

        self._received_weights = {}


    def perform_update(self) -> tuple[ModelWeights, ModelWeights, Loss, int]:
        """
        Perform a model update, training the node model on local data for a given number of epochs.

        The number of epochs is the one specific in the global training configuration object.

        Set the node state to TRAINING and leave it on, in order to stop reception of new models
        until the current one will be saved.

        :return: latest model weights, weights of the best trained model, its loss and the current number of updates without improvements
        """
        self.state = NodeState.TRAINING

        latest_weights, best_weights, best_val_loss = self.train_model(
            n_epochs=self._training_config.epochs_per_update,
        )

        self._completed_updates += 1

        self._evaluate()

        return (
            latest_weights,
            best_weights,
            best_val_loss,
            self._updates_without_improving,
        )

    def train_model(self, n_epochs: int) -> tuple[ModelWeights, ModelWeights, Loss]:
        """
        Train the node model on local data for the specified number of epochs.

        After every epoch, store the training and validation metrics that will be used to build the
        training history. Also, keep track of the best obtained validation loss (among the performed
        epochs) and the related weights and return them.

        :param n_epochs: the number of training epochs.
        :return: latest model weights, weights of the best trained model and best validation loss
        """
        if n_epochs < 1:
            raise Exception("Epochs number must be at least 1!")

        model = self._create_model()
        model.set_weights(self._model.get_weights())

        if self._training_config.use_fedprox:
            if not hasattr(
                model,
                "set_fedprox_reference_weights",
            ):
                raise TypeError(
                    "FedProx is enabled, but the created model "
                    "does not support FedProx."
                )

            model.set_fedprox_reference_weights()


        X_train = self.data["X_train"]
        Y_train = self.data["Y_train"]
        if self._synthetic_data_by_node_type:

            synthetic_x = [
                data[0]
                for data in self._synthetic_data_by_node_type.values()
            ]

            synthetic_y = [
                data[1]
                for data in self._synthetic_data_by_node_type.values()
            ]

            X_train = np.concatenate(
                [X_train] + synthetic_x,
                axis=0,
            )

            Y_train = np.concatenate(
                [Y_train] + synthetic_y,
                axis=0,
            )

        # BLOCCO TEMPORANEO PER TESTARE
        print(
            f"[{self._training_config.merge_strategy}] Node {self.id} training with "
            f"X={X_train.shape}, Y={Y_train.shape}, "
            f"real_samples={len(self.data['X_train'])}, "
            f"synthetic_types={list(self._synthetic_data_by_node_type.keys())}"
        )
        # -----------------------------

        X_prepared, Y_prepared = prepare_janossy_input(
            X_train, Y_train, num_permutations = 6
        )
        X_val_prepared, Y_val_prepared = prepare_janossy_input(
            self.data["X_val"], self.data["Y_val"], num_permutations = 6
        )

        best_val_loss = math.inf
        best_weights = None
        
        for i in range(n_epochs):
            history = model.fit(
                X_prepared,
                Y_prepared,
                epochs=1,
                validation_data=(X_val_prepared, Y_val_prepared),
                verbose=0,
                batch_size=self._training_config.batch_size,
                validation_batch_size=self._training_config.batch_size,
                shuffle=self._training_config.shuffle_batch,
                callbacks=[self._tracker],
            ).history

            if len(model.loss) > 1:
                val_loss = sum([history[f"val_{l}_loss"][0] for l in model.loss.keys()])
            else:
                val_loss = history["val_loss"][0]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights()

            for metric in history:
                metric_value = history[metric][0]

                if metric not in self.training_history:
                    self.training_history[metric] = [metric_value]
                else:
                    self.training_history[metric].append(metric_value)

        assert best_weights
        latest_weights = model.get_weights()

        if hasattr(
            model,
            "clear_fedprox_reference_weights",
        ):
            model.clear_fedprox_reference_weights()

        if (
            self._diagnostics_enabled()
            and getattr(self, "_diagnostics_pending", False)
        ):
            self._diagnostics_evaluate(
                model,
                "after_training",
            )
            self._diagnostics_pending = False

        return latest_weights, best_weights, best_val_loss

    def marshal_model(self) -> WeightsMessage:
        """
        Sample weights from the current model accordingly to the percentage specified in the config.

        :return: the sampled weights
        """
        return WeightsMessage(
            marshaled_weights=self._marshal_weights_fn(
                self._model, self._training_config.perc_sent_weights
            ),
            model_weight=self.accumulated_weight,
            optimizer_state=self._model.optimizer.variables()
            if self._training_config.serialize_optimizer
            else None,
            sender_node_type=self.node_type,
        )

    def save_model(
        self,
        *,
        latest_weights: ModelWeights,
        best_update_model_weights: ModelWeights,
        time: int,
        best_update_val_loss: Loss,
        updates_without_improving: int,
        new_model_weight: int,
    ) -> None:
        """
        Update the best model and reset the early stopping counter if necessary.

        Update the best model with the received model weights, if it improved the validation loss.
        Also, update the best validation loss achieved so far in that case and reset the early
        stopping counter.

        Otherwise, increase the early stopping counter by one.
        Check if the stop criterion is met and eventually change the node state to STOPPED.

        :param best_update_model_weights: the weights of the best model trained during the last update
        :param latest_weights: the weights to be saved.
        :param time: the current time.
        :param best_update_val_loss: the validation loss achieved by the received weights
        :param updates_without_improving: the number of updates without improvements
        :param new_model_weight: the new model weight
        :return: whether the node has improved the best loss
        """
        self._model.set_weights(latest_weights)
        self.accumulated_weight = new_model_weight

        improvement = self._best_val_loss - best_update_val_loss
        reset_early_stopping = improvement >= self._training_config.min_delta

        if improvement > 0:
            self._logger.debug_log(
                f"Node {self.id} improved loss by {improvement:.4f}. Early stopping is"
                f"{'' if reset_early_stopping else ' not'} reset."
            )
        else:
            self._logger.debug_log(f"Node {self.id} did not improve loss")

        self.update_best_model(best_update_model_weights, best_update_val_loss)

        # if the improvement is greater than min_delta, reset early stopping counter; otherwise,
        # increase it and eventually stop the node if the max number of epochs without improving
        # is reached
        if reset_early_stopping:
            self._updates_without_improving = 0
            self._last_improved_time = time
        else:
            self._updates_without_improving = updates_without_improving + 1
            self._logger.debug_log(
                f"This was the {self._updates_without_improving} update without improvement for node {self.id}"
            )

        self._check_stop_criterion()

    def update_best_model(self, weights: ModelWeights, val_loss: float):
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_weights = weights

    def persist_best_model(self) -> None:
        self._model.save(
            str(
                self._workspace_dir
                / self._training_config.models_folder
                / f"{self.id}.keras"
            )
        )
    def receive_weights(self, received: WeightsMessage, from_node: NodeId) -> None:
        """
        Receive marshaled weights from a node and store them in the internal buffer.

        :param received: the received weights
        :param from_node: the node from which the weights came from
        """
        self._received_weights[from_node] = received

    def _evaluate(self) -> None:
        if (
            self._history_config.eval_test
            and self._completed_updates % self._history_config.freq == 0
        ):
            X, Y = self._test_set
            x_test = prepare_janossy_test_input(X, num_permutations=6)
            
            pred = self._model.predict(x_test, verbose=0)

            metrics = compute_metrics(self._test_set[1], pred)
            self.eval_metrics.append(metrics)

    @property
    def ready_to_train(self) -> bool:
        """
        Whether then node has buffered a minimum number of models to perform an update.
        """
        return len(self._received_weights) >= self._training_config.num_merged_models

    def _check_stop_criterion(self):
        """
        Check if the stop criterion is met and eventually stop the node.
        """
        if self._training_config.stop_criterion == StopCriterion.NO_IMPROVEMENTS:
            satisfied_stop_criterion = (
                self._updates_without_improving == self._training_config.patience
            )
        elif self._training_config.stop_criterion == StopCriterion.FIXED_UPDATES:
            satisfied_stop_criterion = (
                self._training_config.fixed_updates == self._completed_updates
            )
        else:
            raise Exception("Unrecognized stop criterion!")

        if (
            satisfied_stop_criterion
            and self._training_config.merge_strategy
            == MergeStrategy.NODE_TYPE_MERGE
        ):
            self.persist_synthetic_data()

        self.state = (NodeState.STOPPED if satisfied_stop_criterion else NodeState.ACTIVE)

