import tensorflow as tf
import tensorflow_probability as tfp


class CommChannel(tf.keras.layers.Layer):
    
    def __init__(self, 
                 size=32,
                 noise=0.5, 
                 temperature=1,
                 no_transform=False,
                 one_hot=True):
        super(CommChannel, self).__init__()
        self.size = size
        self.noise = noise
        self.temperature = temperature
        self.no_transform = no_transform
        self.one_hot = one_hot
        
    def get_initial_state(self, batch_size):
        return tf.zeros((batch_size, self.size))
        
    def call(self, x, training=False):
        if training:
            if self.no_transform:
                return x
            
            if self.noise > 0:
                x = x + tf.random.normal(tf.shape(x),  
                                         mean=tf.zeros_like(x), 
                                         stddev=self.noise)
                
            # The RelaxedOneHotCategorical distribution was concurrently introduced as the
            # Gumbel-Softmax (Jang et al., 2016) and Concrete (Maddison et al., 2016)
            # distributions for use as a reparameterized continuous approximation to the
            # `Categorical` one-hot distribution. If you use this distribution, please cite
            # both papers.
            dist = tfp.distributions.RelaxedOneHotCategorical(
                self.temperature, 
                logits=x
            )
            return dist.sample()
        
        else:
            if self.one_hot:
                x = tf.nn.softmax(x)
                
            x = tf.one_hot(tf.argmax(x, axis=-1), 
                           self.size, dtype=tf.float32)
            return x