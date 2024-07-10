import tensorflow as tf
import numpy as np

class TemporalFusionTransformer(tf.keras.Model):

    # Constructor
    def __init__(self, params):

        super(TemporalFusionTransformer, self).__init__()

        self.num_encoder_steps                  = params['num_encoder_steps']
        self.num_decoder_steps                  = params['num_decoder_steps']
        self.hidden_layer_size                  = params['hidden_layer_size']
        self.num_heads                          = params['num_heads']
        self.dropout_rate                       = params['dropout_rate']
        self.num_static                         = params['num_static']
        self.num_inputs                         = params['num_inputs']
        self.num_outputs                        = params['num_outputs']
        self.category_counts                    = params['category_counts']
        self.quantiles                          = params['quantiles']

        # Input transformations
        self.static_covariates_layer            = tf.keras.layers.Dense(self.hidden_layer_size)
        self.dynamic_covariates_layer           = tf.keras.layers.Dense(self.hidden_layer_size)

        # Embedding layers for categorical variables
        self.static_embedding_layers            = [tf.keras.layers.Embedding(count, self.hidden_layer_size) for count in self.category_counts['static']]
        self.dynamic_embedding_layers           = [tf.keras.layers.Embedding(count, self.hidden_layer_size) for count in self.category_counts['dynamic']]

        # Variable selection networks
        self.static_vsn                         = VariableSelectionNetwork(self.hidden_layer_size, self.num_static)
        self.dynamic_vsn_past                   = VariableSelectionNetwork(self.hidden_layer_size, self.num_inputs)
        self.dynamic_vsn_future                 = VariableSelectionNetwork(self.hidden_layer_size, self.num_inputs)

        # Static context for variable selection
        self.static_context_variable_selection  = GatedResidualNetwork(self.hidden_layer_size)
        self.static_context_enrichment          = GatedResidualNetwork(self.hidden_layer_size)
        self.static_context_state_h             = GatedResidualNetwork(self.hidden_layer_size)
        self.static_context_state_c             = GatedResidualNetwork(self.hidden_layer_size)

        # Temporal fusion decoder
        self.lstm_encoder                       = tf.keras.layers.LSTM(self.hidden_layer_size, return_state=True)
        self.lstm_decoder                       = tf.keras.layers.LSTM(self.hidden_layer_size, return_sequences=True)

        # Self-attention layer
        self.self_attn                          = InterpretableMultiHeadAttention(self.num_heads, self.hidden_layer_size, self.dropout_rate)

        # Gated skip connection
        self.gated_skip_connection              = GatedLinearUnit(self.hidden_layer_size)

        # GLU for temporal feature fusion
        self.temporal_feature_fusion            = GatedLinearUnit(self.hidden_layer_size)

        # Output layer
        self.output_layer                       = tf.keras.layers.Dense(self.num_outputs * len(self.quantiles))

    def call(self, inputs, training=False):

        static_inputs, past_inputs, future_inputs = inputs

        # Process static inputs
        static_embeddings                       = self.process_static_inputs(static_inputs)
        static_encoder, static_weights          = self.static_vsn(static_embeddings)

        # Generate static contexts
        context_var_selection                   = self.static_context_variable_selection(static_encoder)
        context_enrichment                      = self.static_context_enrichment(static_encoder)
        context_state_h                         = self.static_context_state_h(static_encoder)
        context_state_c                         = self.static_context_state_c(static_encoder)

        # Process past and future inputs
        past_embeddings                         = self.process_dynamic_inputs(past_inputs)
        future_embeddings                       = self.process_dynamic_inputs(future_inputs)

        past_transformed, past_flags            = self.dynamic_vsn_past(past_embeddings, context_var_selection)
        future_transformed, future_flags        = self.dynamic_vsn_future(future_embeddings, context_var_selection)

        # LSTM encoding
        lstm_input                              = tf.concat([past_transformed, future_transformed], axis=1)
        lstm_output, state_h, state_c           = self.lstm_encoder(lstm_input, initial_state=[context_state_h, context_state_c])

        # Apply gated skip connection
        temporal_features, _                    = self.gated_skip_connection(lstm_input, lstm_output)

        # Temporal self-attention
        attn_input                              = tf.concat([temporal_features, context_enrichment], axis=-1)
        attn_output, attn_weights               = self.self_attn(attn_input, attn_input, attn_input)

        # Gated feed-forward layer
        decoder_output                          = self.temporal_feature_fusion(attn_output, context_enrichment)

        # Final output layer
        outputs                                 = self.output_layer(decoder_output)

        return outputs

    # Features which are known or predictable in the future, such as `day of week`, `day of month`
    def process_static_inputs(self, static_inputs):

        continuous, categorical                 = static_inputs
        continuous_emb                          = self.static_covariates_layer(continuous)
        categorical_emb                         = [layer(categorical[:, i]) for i, layer in enumerate(self.static_embedding_layers)]
        return tf.concat([continuous_emb] + categorical_emb, axis=-1)

    # Unknown features which are dynamic
    def process_dynamic_inputs(self, dynamic_inputs):

        continuous, categorical                 = dynamic_inputs
        continuous_emb                          = self.dynamic_covariates_layer(continuous)
        categorical_emb                         = [layer(categorical[:, :, i]) for i, layer in enumerate(self.dynamic_embedding_layers)]
        return tf.concat([continuous_emb] + categorical_emb, axis=-1)

class VariableSelectionNetwork(tf.keras.layers.Layer):

    def __init__(self, hidden_size, num_inputs):

        super(VariableSelectionNetwork, self).__init__()
        self.hidden_size                        = hidden_size
        self.num_inputs                         = num_inputs
        self.grn_flat                           = GatedResidualNetwork(hidden_size, output_size=num_inputs)
        self.grn_vars                           = [GatedResidualNetwork(hidden_size) for _ in range(num_inputs)]

    def call(self, inputs, context=None):

        if context is not None:
            flat                                = tf.concat([inputs, context], axis=-1)
        else:
            flat                                = inputs

        weights                                 = tf.nn.softmax(self.grn_flat(flat), axis=-1)
        transformed_inputs                      = tf.stack([self.grn_vars[i](inputs[:, :, i]) for i in range(self.num_inputs)], axis=-1)

        outputs                                 = tf.reduce_sum(transformed_inputs * tf.expand_dims(weights, axis=-1), axis=-1)

        return outputs, weights

class GatedResidualNetwork(tf.keras.layers.Layer):

    # Constructor
    def __init__(self, hidden_size, output_size=None, dropout_rate=0.1):

        super(GatedResidualNetwork, self).__init__()

        self.hidden_size                        = hidden_size
        self.output_size                        = output_size or hidden_size
        self.dropout_rate                       = dropout_rate

        self.layer_norm                         = tf.keras.layers.LayerNormalization()
        self.dense1                             = tf.keras.layers.Dense(hidden_size, activation='elu')
        self.dense2                             = tf.keras.layers.Dense(hidden_size, activation='elu')
        self.dense3                             = tf.keras.layers.Dense(self.output_size)
        self.gate                               = tf.keras.layers.Dense(self.output_size, activation='sigmoid')
        self.dropout                            = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, context=None, training=False):

        x                                       = self.layer_norm(inputs)

        if context is not None:
            x                                   = tf.concat([x, context], axis=-1)

        x                                       = self.dense1(x)
        x                                       = self.dropout(x, training=training)
        x                                       = self.dense2(x)
        x                                       = self.dropout(x, training=training)
        x                                       = self.dense3(x)
        gating                                  = self.gate(inputs)

        return x * gating + inputs

class InterpretableMultiHeadAttention(tf.keras.layers.Layer):

    # Constructor
    def __init__(self, num_heads, d_model, dropout_rate):

        super(InterpretableMultiHeadAttention, self).__init__()

        self.num_heads                          = num_heads
        self.d_model                            = d_model
        self.depth                              = d_model // self.num_heads

        self.wq                                 = tf.keras.layers.Dense(d_model)
        self.wk                                 = tf.keras.layers.Dense(d_model)
        self.wv                                 = tf.keras.layers.Dense(d_model)

        self.dense                              = tf.keras.layers.Dense(d_model)
        self.dropout                            = tf.keras.layers.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):

        x                                       = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):

        batch_size                              = tf.shape(q)[0]

        q                                       = self.wq(q)
        k                                       = self.wk(k)
        v                                       = self.wv(v)

        q                                       = self.split_heads(q, batch_size)
        k                                       = self.split_heads(k, batch_size)
        v                                       = self.split_heads(v, batch_size)

        scaled_attention, attention_weights     = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention                        = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention                        = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output                                  = self.dense(concat_attention)
        output                                  = self.dropout(output)

        return output, attention_weights

class GatedLinearUnit(tf.keras.layers.Layer):

    # Contructor
    def __init__(self, hidden_size):

        super(GatedLinearUnit, self).__init__()

        self.hidden_size                        = hidden_size
        self.dense                              = tf.keras.layers.Dense(hidden_size * 2)

    def call(self, inputs, context=None):

        if context is not None:
            x                                   = tf.concat([inputs, context], axis=-1)
        else:
            x                                   = inputs

        x                                       = self.dense(x)

        output, gating                          = tf.split(x, 2, axis=-1)

        return output * tf.nn.sigmoid(gating)

def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk                                   = tf.matmul(q, k, transpose_b=True)
    dk                                          = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits                     = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits                 += (mask * -1e9)

    attention_weights                           = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output                                      = tf.matmul(attention_weights, v)

    return output, attention_weights

def quantile_loss(y_true, y_pred, quantiles):
    q                                           = tf.constant(np.array(quantiles), dtype=tf.float32)
    e                                           = y_true - y_pred
    v                                           = tf.maximum(q * e, (q - 1) * e)
    return tf.reduce_mean(v)

# Parameters
params = {
    'num_encoder_steps' : 24,
    'num_decoder_steps' : 12,
    'hidden_layer_size' : 64,
    'num_heads'         :  4,
    'dropout_rate'      :  0.1,
    'num_static'        :  5,
    'num_inputs'        : 10,
    'num_outputs'       :  1,
    'category_counts'   :  {
                            'static'    : [3, 4],
                            'dynamic'   : [5, 2]
                           },
    'quantiles'         : [0.1, 0.5, 0.9]
}

# Initialize a model
model       = TemporalFusionTransformer(params)

# Compile the model
optimizer   = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: quantile_loss(y_true, y_pred, params['quantiles']))

# Train the model
#model.fit( [static_inputs, past_inputs, future_inputs], y_train, epochs=50, batch_size=32 )

# Make predictions with trained model
#predictions = model.predict( [static_inputs_test, past_inputs_test, future_inputs_test] )
