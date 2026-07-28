import math
import numpy as np
from enum import IntEnum
from pathlib import Path
from typing import Optional

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
        self._synthetic_data_by_node_type: dict[int, tuple[np.ndarray, np.ndarray]] = {}

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
        if new_node_type not in (0, 1, 2):
            raise ValueError(
                f"Invalid node_type {new_node_type}. Expected 0, 1 or 2."
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

        if received_node_type not in (0, 1, 2):
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

    def persist_synthetic_data(self) -> None:
        """
        Save the synthetic datasets generated by this node.

        One compressed NPZ file is created for each remote node type.
        """
        synthetic_dir = (
            self._workspace_dir
            / "synthetic_data"
            / f"node_{self.id}"
        )

        synthetic_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for node_type, (
            X_synthetic,
            Y_synthetic,
        ) in self._synthetic_data_by_node_type.items():

            output_file = (
                synthetic_dir
                / f"node_type_{node_type}.npz"
            )

            np.savez_compressed(
                output_file,
                X=X_synthetic,
                Y=Y_synthetic,
            )

        print(
            f"[SYNTHETIC_SAVE] Node {self.id}: "
            f"saved types="
            f"{list(self._synthetic_data_by_node_type.keys())} "
            f"in {synthetic_dir}"
        )


    def load_synthetic_data(
        self,
        source_dir: Path,
    ) -> None:
        """
        Load synthetic datasets generated for this node
        during a previous NODE_TYPE_MERGE run.
        """
        node_dir = source_dir / f"node_{self.id}"

        if not node_dir.exists():
            raise FileNotFoundError(
                f"Synthetic data directory not found "
                f"for node {self.id}: {node_dir}"
            )

        loaded_types: list[int] = []

        for data_file in sorted(
            node_dir.glob("node_type_*.npz")
        ):
            node_type_string = data_file.stem.replace(
                "node_type_",
                "",
            )

            try:
                node_type = int(node_type_string)
            except ValueError as error:
                raise RuntimeError(
                    f"Invalid synthetic data filename: "
                    f"{data_file.name}"
                ) from error

            if node_type not in (0, 1, 2):
                raise RuntimeError(
                    f"Invalid synthetic node type "
                    f"{node_type} in {data_file}"
                )

            if node_type == self.node_type:
                raise RuntimeError(
                    f"Node {self.id} has local type "
                    f"{self.node_type}, but the loaded synthetic "
                    f"dataset has the same node type."
                )

            with np.load(
                data_file,
                allow_pickle=False,
            ) as data:
                if "X" not in data or "Y" not in data:
                    raise RuntimeError(
                        f"File {data_file} must contain "
                        "'X' and 'Y' arrays."
                    )

                X_synthetic = data["X"]
                Y_synthetic = data["Y"]

            if X_synthetic.ndim != 3:
                raise RuntimeError(
                    f"Invalid synthetic X in {data_file}: "
                    f"expected 3 dimensions, received "
                    f"{X_synthetic.shape}"
                )

            if Y_synthetic.ndim != 2:
                raise RuntimeError(
                    f"Invalid synthetic Y in {data_file}: "
                    f"expected 2 dimensions, received "
                    f"{Y_synthetic.shape}"
                )

            if len(X_synthetic) != len(Y_synthetic):
                raise RuntimeError(
                    f"Different numbers of samples in "
                    f"{data_file}: "
                    f"X={len(X_synthetic)}, "
                    f"Y={len(Y_synthetic)}"
                )

            if Y_synthetic.shape[1] != 4:
                raise RuntimeError(
                    f"Synthetic Y in {data_file} must "
                    f"contain 4 columns. Received shape "
                    f"{Y_synthetic.shape}."
                )

            expected_x_shape = self.data["X_train"].shape[1:]

            if X_synthetic.shape[1:] != expected_x_shape:
                raise RuntimeError(
                    f"Invalid synthetic X shape in {data_file}. "
                    f"Expected (*, {expected_x_shape}), "
                    f"received {X_synthetic.shape}."
                )

            self._synthetic_data_by_node_type[node_type] = (
                X_synthetic,
                Y_synthetic,
            )

            loaded_types.append(node_type)

        if not loaded_types:
            raise RuntimeError(
                f"No synthetic data files found for "
                f"node {self.id} in {node_dir}"
            )

        print(
            f"[SYNTHETIC_LOAD] Node {self.id}: "
            f"loaded types={loaded_types} "
            f"from {node_dir}"
        )

    def merge_models(self) -> None:
        """
        Merge all the received model weights into the current model.

        The internal model weights are updated. The number of trained samples is set at the
        maximum between the number of trained samples of the merged models.

        With NODE_TYPE_MERGE, each received model is first used to generate
        synthetic data representing its node type. Synthetic data for the
        same node type replace the previously stored block.
        """

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

                # The dictionary key is the received node type.
                # A new block automatically replaces the old block
                # belonging to the same node type.
                self._synthetic_data_by_node_type[received_node_type] = (
                    X_synthetic,
                    Y_synthetic,
                )

                #BLOCCO TEMPORANEO PER TESTARE
                print(
                    f"[{self._training_config.merge_strategy}] Node {self.id} "
                    f"(type={self.node_type}) received type={received_node_type}; "
                    f"synthetic X={X_synthetic.shape}, "
                    f"synthetic Y={Y_synthetic.shape}; "
                    f"stored types={list(self._synthetic_data_by_node_type.keys())}"
                )
                # -----------------------------

        self._model, self.accumulated_weight = self._aggregator(
            self._model,
            self.accumulated_weight,
            tuple(msg for k, msg in self._received_weights.items()),
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

