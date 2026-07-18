import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        self.r = self.r_act.forward(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)
        self.z = self.z_act.forward(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        
        self.n = self.h_act.forward(self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh))

        h_t = (1 - self.z) * self.n + self.z * h_prev_t

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        """GRU cell backward."""
        
        # 1. Gradients from the final output h_t = (1 - z) * n + z * h_prev
        dn = delta * (1 - self.z)
        dz = delta * (self.hidden - self.n)
        dh_prev_direct = delta * self.z

        # 2. Backprop through the Tanh activation for the candidate (n)
        # Using the activation's backward method instead of manual calculation
        da_n = self.h_act.backward(dn)

        # 3. Calculate gradients related to the reset gate's interaction
        # Reconstruct the pre-multiplication state from the forward pass
        pre_n_h = self.Wnh @ self.hidden + self.bnh
        
        # Gradient wrt the reset gate (r)
        dr = da_n * pre_n_h
        
        # Gradient wrt the linear transformation (Wnh @ h_prev + bnh)
        dpre_n_h = da_n * self.r

        # 4. Backprop through the Sigmoid activations for the gates (r and z)
        # Using the activation's backward method
        da_r = self.r_act.backward(dr)
        da_z = self.z_act.backward(dz)

        # 5. Reshape vectors to column vectors for matrix outer products (dWs)
        x_col = self.x.reshape(-1, 1)        # (d, 1)
        h_col = self.hidden.reshape(-1, 1)   # (h, 1)
        
        da_n_col = da_n.reshape(-1, 1)       # (h, 1)
        da_r_col = da_r.reshape(-1, 1)       # (h, 1)
        da_z_col = da_z.reshape(-1, 1)       # (h, 1)
        dpre_n_h_col = dpre_n_h.reshape(-1, 1) # (h, 1)

        # 6. Calculate weight and bias gradients
        # Candidate (n) weights and biases
        self.dWnx = da_n_col @ x_col.T
        self.dWnh = dpre_n_h_col @ h_col.T
        self.dbnx = da_n
        self.dbnh = dpre_n_h

        # Reset gate (r) weights and biases
        self.dWrx = da_r_col @ x_col.T
        self.dWrh = da_r_col @ h_col.T
        self.dbrx = da_r
        self.dbrh = da_r

        # Update gate (z) weights and biases
        self.dWzx = da_z_col @ x_col.T
        self.dWzh = da_z_col @ h_col.T
        self.dbzx = da_z
        self.dbzh = da_z

        # 7. Calculate gradients wrt inputs (dx and dh_prev_t)
        # dx is the sum of gradients flowing back through Wnx, Wrx, and Wzx
        dx = self.Wnx.T @ da_n + self.Wrx.T @ da_r + self.Wzx.T @ da_z
        
        # dh_prev_t is the sum of the direct flow + flow through Wnh, Wrh, and Wzh
        dh_prev_t = dh_prev_direct + self.Wnh.T @ dpre_n_h + self.Wrh.T @ da_r + self.Wzh.T @ da_z

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t
