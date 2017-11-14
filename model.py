import tensorflow as tf
import ops

BATCH_SIZE=32

class GAN():
    """ 
    Wasserstein Generative Adversarial Network with Gradient Penalty 
    and Auxiliary Classifier
    (WGAN-GP-AC)
    """
    def __init__(self, n_classes=0, n_latent=128, multiclass=False):
        """
        Args:
            n_classes   (int): Number of classes if using auxiliary classifier
            n_latent    (int): Latent vector Z
            multiclass (bool): False if classes are mutually exclusive
        """
        # Member variables
        self.n_classes, self.n_latent, self.multiclass = n_classes, n_latent, multiclass
        # Hyper parameters
        self.Lambda = 10

        self.G = Generator(n_latent)
        self.C = Critic(n_classes)

    def build_train_graph(self, batch_size, examples_real, labels_real=None):
        # TODO: add label smoothing
        z = self.random_latent(batch_size)
        # Not using auxiliary classifier
        if labels_real is None: 
            examples_fake = self.G.generate(z)

            scores_real = self.C.critique(examples_real)
            scores_fake = self.C.critique(examples_fake, reuse=True)
        # Using auxiliary classifier
        else:                       
            labels_fake = ops.random_labels(batch_size, self.n_classes, self.multiclass)
            examples_fake = self.G.generate(z, labels_fake)

            scores_real, logits_real = self.C.critique(examples_real)
            scores_fake, logits_fake = self.C.critique(examples_fake, reuse=True)
        
        tf.summary.image('examples_real', examples_real)
        tf.summary.image('examples_fake', examples_fake)
        
        with tf.name_scope('loss'):
            critic_loss, gen_loss = self.wgan_loss(scores_real, scores_fake)
            critic_loss, gen_loss = [critic_loss], [gen_loss]
            # using auxiliary classifier
            if labels_real is not None:
                # Add loss from auxiliary classifier
                ac_critic_loss, ac_gen_loss = self.ac_loss(
                        labels_real, labels_fake, logits_real, logits_fake)
                critic_loss.append(ac_critic_loss)
                gen_loss.append(ac_gen_loss)

            gradient_penalty = self.gp_loss(examples_real, examples_fake)
            critic_loss.append(gradient_penalty)

            critic_loss = tf.add_n(critic_loss)
            gen_loss = tf.add_n(gen_loss)
            tf.summary.scalar('loss', gen_loss)

            critic_fetch = (self.C.train_op(critic_loss), critic_loss)
            gen_fetch = (self.G.train_op(gen_loss), gen_loss)
        return critic_fetch, gen_fetch
        

    def random_latent(self, batch_size):
        z = tf.random_normal([batch_size, self.n_latent], stddev=1.0, name='Z')
        return z
    
    def gp_loss(self, examples_real, examples_fake):
        """
        Gradient Penalty from ("Improved Training of Wasserstein GANs")
        [https://arxiv.org/abs/1704.00028]

        Args:
            examples_real (Tensor[batch_size, width, height, channels]):
            examples_fake (Tensor[batch_size, width, height, channels]):
        Returns: Tensor: scalar penalty for maintaining lipschitz constraint: """
        with tf.name_scope('gradient_penalty'):
            batch_size = tf.shape(examples_real)[0]
            alpha = tf.random_uniform(shape=[batch_size,1,1,1], dtype=tf.float32)

            differences = examples_fake - examples_real
            interpolates = examples_real + (alpha * differences)
            
            result = self.C.critique(interpolates, reuse=True)
            # May be using auxilary classifier
            scores_mixed = result[0] if type(result) is tuple else result

            gradients = tf.gradients(scores_mixed, [interpolates])[0]

            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[-1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            critic_gp = self.Lambda*gradient_penalty
            return critic_gp

    def drift_loss(self):
        # TODO: prevents critic score real from drifting away from zero
        pass

    def wgan_loss(self, scores_real, scores_fake):
        """
        Args:
            scores_fake (Tensor[batch_size, 1]):
            scores_real (Tensor[batch_size, 1]):
        Returns:
            Tensor: critic loss (scalar)
            Tensor: generator loss (scalar)
        """
        with tf.name_scope('wasserstein_loss'):
            loss_real = tf.reduce_mean(scores_real)
            loss_fake = tf.reduce_mean(scores_fake)

            generator_loss = -loss_fake
            critic_loss = loss_fake - loss_real
        
            return critic_loss, generator_loss

    def ac_loss(self, labels_real, labels_fake, 
            logits_real, logits_fake):
        """
        Loss from the auxilliary classifier
        Args:
        logits_* (Tensor): output of the critic axiliary classifier
            with shape [batch_size, num_classes]
        labels_* (Tensor): corresponding labels
            with shape [batch_size, num_classes]
        Returns:
        Tensor: scalar critic loss
        Tensor: scalar generator loss
        """
        cross_entropy = tf.losses.sigmoid_cross_entropy \
               if self.multiclass else tf.losses.softmax_cross_entropy

        with tf.name_scope('ac_loss'):
            generator_loss = cross_entropy(
                    labels_fake, logits_fake, loss_collection=None) # losses_collection?
            critic_loss = generator_loss + \
                    cross_entropy(labels_real, logits_real, loss_collection=None)

            return critic_loss, generator_loss

class _Model():
    learning_rate=1e-4
    beta1=0.5
    beta2=0.9
    name=None   # Either 'Critic' or 'Generator'

    def trainable(self):
        return tf.trainable_variables(scope=self.name)


    def train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(
                self.learning_rate,
                self.beta1,
                self.beta2)
        return optimizer.minimize(loss, var_list=self.trainable())

class Generator(_Model):
    name='Generator'
    def __init__(self, n_latent=128):
        self.n_latent = n_latent

    def generate(self, z, labels=None):
        """
        Forward pass of the generator
        Args:
            z      (Tensor): latent vector
            labels (Tensor): optional labels for conditional generator
        """
        layer = z if labels is None else tf.concat([labels,z], -1)

        with tf.variable_scope('Generator'):
            layer = ops.dense(layer, 4*4*4*64, name='dense')
            layer = tf.reshape(layer, [-1, 4, 4, 4*64])
            # 4x4x256
            layer = ops.conv2d_transpose(layer, 2*64, strides=2, name='convT1')
            # 8x8x128
            layer = layer[:,:7,:7,:]
            # 7x7x128
            layer = ops.conv2d_transpose(layer, 64, strides=2, name='convT2')
            # 14x14x64
            layer = ops.conv2d_transpose(layer, 1, strides=2, 
                    activation=tf.nn.sigmoid, use_bias=True, name='convT3')
            # 28x28x1
            return layer

class Critic(_Model):
    name='Critic'
    def __init__(self, n_classes=0):
        self.n_classes = n_classes

    def critique(self, examples, reuse=False):
        """
        Forward pass of the discriminator/critic
        Args:
            examples (Tensor): shape is [batch_size, width, height, channels]
        """
        with tf.variable_scope('Critic', reuse=reuse):
            # 28x28x1
            layer = ops.conv2d(examples, 64, name='conv1', strides=2)
            # 14x14x64
            layer = ops.conv2d(layer, 2*64, name='conv2', strides=2)
            # 7x7x128
            layer = ops.conv2d(layer, 4*64, name='conv3', strides=2)
            # 4x4x256
            layer = tf.reshape(layer, [-1, 4*4*4*64])
            layer = ops.dense(layer, self.n_classes+1, use_bias=True, name='linear')

            # First element of each batch is the critic score
            # The rest of the elements are the logits for the auxiliary classifier
            critiques = layer[:, 0]
            if self.n_classes > 0:
                logits = layer[:,1:]
                return critiques, logits
            return critiques
