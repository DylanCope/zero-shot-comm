import tensorflow as tf


class Agent(tf.keras.Model):
    
    def __init__(
        self, 
        channel_size,
        num_classes,
        name='agent',
        lstm_units=64,
        dense_dim=128,
        encoder=None,
        output_activation=None,
        first_activation='relu',
        unknown_class=False,
        dropout_prob=0.,
        **kwargs
    ):
        super(Agent, self).__init__(name=name, **kwargs)
        self.channel_size = channel_size
        self.num_classes = num_classes
        
        self.concat = tf.keras.layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(dropout_prob)
        
        self.encoder = encoder or \
            tf.keras.layers.Lambda(lambda x: x)
        
        self.dense1 = tf.keras.layers.Dense(
            128, activation=first_activation
        )
        
#         self.dense2 = tf.keras.layers.Dense(
#             64, activation='relu'
#         )
        
        self.lstm = tf.keras.layers.LSTM(
            units=64,
            return_state=True,
            dropout=dropout_prob,
        )

        self.unknown_class = unknown_class
        if self.unknown_class:
            output_size = self.channel_size + self.num_classes + 1
        else:
            output_size = self.channel_size + self.num_classes
        
#         self.extract_utterance = tf.keras.layers.Dense(
#             channel_size, activation=None
#         )
        
#         self.extract_class_probs = tf.keras.layers.Dense(
#             num_classes, activation='softmax'
#         )

        
        self.output_layer = tf.keras.layers.Dense(
            output_size, activation=output_activation
        )
        self.extract_utterance = tf.keras.layers.Lambda(
            lambda x: x[:, :self.channel_size]
        )
        self.extract_class_probs = tf.keras.layers.Lambda(
            lambda x: tf.nn.softmax(x[:, self.channel_size:])
        )

    def call(self, inputs, state=None, training=False):
        
        inp, prev_utt, other_utt = inputs
        
        x = self.encoder(inp)
        x = self.concat([x, prev_utt, other_utt])
        
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        
#         x = self.dense2(x)
        
        x = tf.expand_dims(x, 1) # Add time dim
        if state is None:
            state = self.lstm.get_initial_state(inputs)
        x, *state = self.lstm(x, initial_state=state, 
                              training=training)
        self.state = state
        
        x = self.output_layer(x)
        utterance = self.extract_utterance(x)
        class_probs = self.extract_class_probs(x)
        
        return utterance, class_probs, state
