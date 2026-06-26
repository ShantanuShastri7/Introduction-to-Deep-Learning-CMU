import numpy as np
import sys

sys.path.append("mytorch")
from rnn_cell import *
from nn.linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO: Understand then uncomment this code :)
        self.rnn = [
            RNNCell(input_size, hidden_size) if i == 0 
                else RNNCell(hidden_size, hidden_size)
                    for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.
        -----
        Input
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]
        linear_weights:
                        [W, b]
        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.
        -----
        Input
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified
        -------
        Returns
        logits: (batch_size, output_size) 

        Output (y): logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None

        for t in range(seq_len):
            input_t = x[:, t, :]  # The input at this time step
        
            for layer_idx in range(self.num_layers):
                h_prev = hidden[layer_idx]
                
                # Pass input and hidden to the cell
                # The cell returns the new hidden state (which is also the output)
                new_h = self.rnn[layer_idx].forward(input_t, h_prev)
                
                # Update the hidden state tracker and the input for the next layer
                hidden[layer_idx] = new_h
                input_t = new_h 
            
            # After passing through all layers, save the final hidden states for this time step
            self.hiddens.append(hidden.copy())

        final_state = self.hiddens[-1][self.num_layers - 1]
        logits = self.output_layer.forward(final_state)

        return logits

    def backward(self, delta):
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        
        # Initialize dh. This will act as our running gradient matrix.
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        
        # The gradient from the output layer enters at the final time step
        # at the uppermost layer (index -1)
        dh[-1] = self.output_layer.backward(delta)

        for t in range(seq_len - 1, -1, -1):
            for l in range(self.num_layers - 1, -1, -1):
                
                # Extract the required forward states
                h_t = self.hiddens[t+1][l]
                h_prev_t = self.hiddens[t][l]
                h_prev_l = self.x[:, t, :] if l == 0 else self.hiddens[t+1][l-1]
                
                # Get gradients from the current cell
                dx, dh_prev = self.rnn[l].backward(dh[l], h_t, h_prev_l, h_prev_t)
                
                # Update dh[l] with the temporal gradient (flowing to t-1)
                dh[l] = dh_prev
                
                # Add dx to the gradient for the layer below (flowing to l-1)
                if l != 0:
                    dh[l-1] += dx

        # After the time loop finishes, dh holds the gradients that reached t=0,
        # which represents the gradient with respect to the initial hidden states (h_0).
        return dh / batch_size