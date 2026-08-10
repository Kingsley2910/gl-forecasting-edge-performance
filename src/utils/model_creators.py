from gossiplearning.config import Config

from tensorflow import keras
from keras.layers import Dropout
from tensorflow.keras import Model, Sequential, Input, layers
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow import reduce_mean

# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.callbacks import CSVLogger

def create_LSTM(config: Config) -> Model: # NON PRESENTE NEL FILE DI KAN 
    optz = Adam(learning_rate=0.001, epsilon=1e-6)

    input_timesteps = 4

    inputs = Input(shape=(input_timesteps, config.training.n_input_features))

    lstm_layers = Sequential(
        [
            LSTM(
                50,
                activation="tanh",
                return_sequences=True,
            ),
            LSTM(
                50,
                activation="tanh",
                return_sequences=False,
            ),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
        ]
    )(inputs)

    outputs = [
        Dense(1, activation="relu", name=f"fn_{i}")(lstm_layers)
        for i in range(config.training.n_output_vars)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optz,
        loss={f"fn_{i}": "mse" for i in range(config.training.n_output_vars)},
        metrics=["mae", "msle", "mse", "mape", RootMeanSquaredError()],
    )

    return model


@register_keras_serializable()
def r2_score_training(y_true, y_pred): # KAN USA LA VERSIONE MASKED
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

@register_keras_serializable() # usato in train_model_cv e train_model (build_janossy_rnn_model)
def masked_r2(y_true, y_pred):
    mask = ~tf.reduce_all(tf.math.is_nan(y_true), axis=-1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + 1e-7)

@register_keras_serializable() # usato in train_model_cv e train_model (build_janossy_rnn_model)
def masked_mae(y_true, y_pred):
    mask = ~tf.reduce_all(tf.math.is_nan(y_true), axis=-1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

@register_keras_serializable()
def masked_mse(y_true, y_pred):
    # Mask samples where ALL regression targets are nan
    sample_mask = ~tf.reduce_all(tf.math.is_nan(y_true), axis=-1)

    # Select only unmasked samples
    y_true_masked = tf.boolean_mask(y_true, sample_mask)
    y_pred_masked = tf.boolean_mask(y_pred, sample_mask)

    return tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))

@register_keras_serializable()
def masked_mape(y_true, y_pred):
    sample_mask = ~tf.reduce_all(tf.math.is_nan(y_true), axis=-1)
    y_true_masked = tf.boolean_mask(y_true, sample_mask)
    y_pred_masked = tf.boolean_mask(y_pred, sample_mask)
    epsilon = tf.keras.backend.epsilon()
    return tf.reduce_mean(tf.abs((y_true_masked - y_pred_masked) / (y_true_masked + epsilon))) * 100

@register_keras_serializable()
class JanossyPooling(layers.Layer): # usato in train_model_cv e train_model (build_janossy_rnn_model)
    def call(self, inputs):
        return reduce_mean(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

@register_keras_serializable()
class FedProxModel(keras.Model):
    """
    Keras model supporting an optional FedProx proximal term.

    The reference weights are fixed before local training and remain
    unchanged for all local epochs belonging to the same gossip update.
    """

    def __init__(
        self,
        *args,
        use_fedprox: bool = False,
        fedprox_mu: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if fedprox_mu < 0:
            raise ValueError(
                "fedprox_mu must be greater than or equal to 0."
            )

        self.use_fedprox = bool(use_fedprox)
        self.fedprox_mu = float(fedprox_mu)

        # Transient reference weights.
        # They are set by the gossip Node before local training.
        self._fedprox_reference_weights = None

    def set_fedprox_reference_weights(self) -> None:
        """
        Store a frozen copy of the current trainable variables.

        This method must be called once after loading/merging the model
        weights and before starting the local epochs.
        """
        if not self.use_fedprox:
            self._fedprox_reference_weights = None
            return

        self._fedprox_reference_weights = [
            tf.stop_gradient(tf.identity(variable))
            for variable in self.trainable_variables
        ]

    def clear_fedprox_reference_weights(self) -> None:
        """
        Remove the reference weights after local training.
        """
        self._fedprox_reference_weights = None

    def _compute_fedprox_penalty(self) -> tf.Tensor:
        """
        Compute:

            mu / 2 * sum_j ||w_j - w_ref_j||^2
        """
        if (
            not self.use_fedprox
            or self.fedprox_mu == 0.0
            or self._fedprox_reference_weights is None
        ):
            return tf.constant(0.0, dtype=tf.float32)

        if len(self.trainable_variables) != len(
            self._fedprox_reference_weights
        ):
            raise RuntimeError(
                "The number of current trainable variables does not "
                "match the number of FedProx reference variables."
            )

        squared_distances = [
            tf.reduce_sum(
                tf.square(
                    current_variable
                    - tf.cast(
                        reference_variable,
                        current_variable.dtype,
                    )
                )
            )
            for current_variable, reference_variable in zip(
                self.trainable_variables,
                self._fedprox_reference_weights,
            )
        ]

        if not squared_distances:
            return tf.constant(0.0, dtype=tf.float32)

        total_squared_distance = tf.add_n(squared_distances)

        return (
            tf.cast(
                self.fedprox_mu,
                total_squared_distance.dtype,
            )
            * total_squared_distance
            / 2.0
        )

    def train_step(self, data):
        """
        Perform one training step using the normal multitask loss
        plus the optional FedProx proximal penalty.
        """
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(
                x,
                training=True,
            )

            # Computes the compiled multitask loss:
            # 5.0 * masked_mse + 1.0 * binary_crossentropy
            task_loss = self.compute_loss(
                x=x,
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                training=True,
            )

            fedprox_penalty = self._compute_fedprox_penalty()

            total_loss = task_loss + fedprox_penalty

        gradients = tape.gradient(
            total_loss,
            self.trainable_variables,
        )

        gradient_variable_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                self.trainable_variables,
            )
            if gradient is not None
        ]

        self.optimizer.apply_gradients(
            gradient_variable_pairs
        )

        # Updates fn_0 metrics and fn_1 metrics separately.
        results = self.compute_metrics(
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )

        results["fedprox_penalty"] = fedprox_penalty
        results["total_loss_with_fedprox"] = total_loss

        return results
    
    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "use_fedprox": self.use_fedprox,
                "fedprox_mu": self.fedprox_mu,
            }
        )

        return config

def build_janossy_rnn_model(config: Config): # usato in train_model_cv e train_model
    """
    Builds a Janossy pooling model using GRU/LSTM and averaging over k permutations.

    Args:
        output_dim: Output dimension (e.g. 1 for regression)
        input_dim: Number of features per timestep (e.g. 8)
        rnn_type: 'gru' or 'lstm'
        rnn_units: Units in the recurrent layer
        num_permutations: Number of permutations per input sequence
    Returns:
        Compiled Keras model
    """
    rnn_type = 'gru'
    rnn_units = 80
    reg_output_activation = "linear"
    reg_loss_function = masked_mse
    cls_output_activation = "sigmoid"
    cls_loss_function = "binary_crossentropy"
    reg_output_dim = config.training.reg_output_dim
    input_dim = config.training.input_dim
    num_permutations= 6 #config.training.num_permutations

    # Input layer
    input_layer = keras.Input(shape=(num_permutations, None, input_dim), name="input_layer")

    # Mask padded zeros before processing
    x = layers.TimeDistributed(
        layers.Masking(mask_value=0.0),
        name="masking_layer"
    )(input_layer)

    # h(x): Embed each timestep feature
    x = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(64, activation='relu', name="dense_embedding")
        ),
        name="time_distributed_embedding"
    )(x)  # -> (batch_size, num_permutations, sequence_length, 64)

    # f*: RNN across each permutation
    if rnn_type.lower() == 'gru':
        x = layers.TimeDistributed(
            layers.GRU(rnn_units, name="gru_rnn"),
            name="time_distributed_rnn"
        )(x)
    else:
        x = layers.TimeDistributed(
            layers.LSTM(rnn_units, name="lstm_rnn"),
            name="time_distributed_rnn"
        )(x)
    # -> (batch_size, num_permutations, rnn_units)

    x = JanossyPooling(name="janossy_pooling")(x) # -> (batch_size, rnn_units)

    # ρ: MLP for final prediction
    x = layers.Dense(128, activation='tanh', name="dense_final")(x)
    x = layers.Dropout(0.2, name="dropout_final")(x)

    # -----------------------------
    # Multi-task heads
    # -----------------------------
    # ----- Regression head -----
    reg_adapter = layers.Dense(128, activation='relu', name='regression_dense')(x)
    reg_adapter = layers.Dropout(0.2, name='regression_dropout')(reg_adapter)
    reg_output = layers.Dense(reg_output_dim, activation=reg_output_activation, name="fn_0")(reg_adapter)

    # ----- Classification head -----
    cls_adapter = layers.Dense(128, activation='relu', name='classification_dense')(x)
    cls_adapter = layers.Dropout(0.2, name='classification_dropout')(cls_adapter)
    cls_output = layers.Dense(1, activation=cls_output_activation, name="fn_1")(cls_adapter)

    # -----------------------------
    # Build and compile model
    # -----------------------------
    if config.training.use_fedprox:
        model = FedProxModel(
            inputs=input_layer,
            outputs=[reg_output, cls_output],
            name="janossy_multitask_model",
            use_fedprox=True,
            fedprox_mu=config.training.fedprox_mu,
        )
    else:
        model = keras.Model(
            inputs=input_layer,
            outputs=[reg_output, cls_output],
            name="janossy_multitask_model",
        )

    model.compile(
        optimizer='adam',
        loss={
            'fn_0': reg_loss_function,
            'fn_1': cls_loss_function
        },
        loss_weights={
            'fn_0': 5.0,
            'fn_1':1.0
        },
        metrics={
            'fn_0': [masked_mae, masked_mse, masked_mape, masked_r2],
            'fn_1': ['accuracy']
        }
    )

    return model

