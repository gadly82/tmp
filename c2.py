import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# Define the RL agent
class RLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize the Transformer model
        self.model = nn.TransformerBlock(
            num_heads=4,
            d_model=128,
            d_ff=256,
        )

        # Initialize the optimizer
        self.optimizer = optax.adam(learning_rate=learning_rate).create(self.model)

    def predict(self, state):
        # Perform forward pass through the model
        logits = self.model(jnp.array(state))
        return jax.nn.softmax(logits)

    def train(self, states, actions, rewards):
        def loss_fn(params, states, actions, rewards):
            logits = self.model(jnp.array(states))
            action_probs = jax.nn.softmax(logits)
            selected_probs = jnp.sum(action_probs * actions, axis=-1)
            loss = -jnp.mean(jnp.log(selected_probs) * rewards)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(self.optimizer.target, states, actions, rewards)
        updates, new_optimizer_state = self.optimizer.update(grad, self.optimizer.state)
        self.optimizer = self.optimizer.replace(state=new_optimizer_state)
        return loss

    def save_model(self, path):
        with open(path, "wb") as f:
            jax.pickle.dump(self.model, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            self.model = jax.pickle.load(f)
