import numpy as np
from mnist import MNIST

# -----------------------------
# --- Load MNIST -------------
# -----------------------------
mndata = MNIST('./mnist_data')
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

X_train = np.array(X_train, dtype=np.float32).reshape(-1,28,28,1)/255.0
X_test = np.array(X_test, dtype=np.float32).reshape(-1,28,28,1)/255.0
y_train = np.array(y_train)
y_test = np.array(y_test)

def one_hot(y, num_classes=10):
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1
    return oh

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# -----------------------------
# --- Activation functions ---
# -----------------------------
def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    return (x>0).astype(np.float32)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# -----------------------------
# --- im2col utilities -------
# -----------------------------
def im2col(X, kh, kw, stride=1):
    N,H,W,C = X.shape
    out_h = (H - kh)//stride + 1
    out_w = (W - kw)//stride + 1
    shape = (N, out_h, out_w, kh, kw, C)
    strides = (X.strides[0], X.strides[1]*stride, X.strides[2]*stride,
               X.strides[1], X.strides[2], X.strides[3])
    cols = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return cols.reshape(N*out_h*out_w, -1)  # (N*out_h*out_w, kh*kw*C)

# -----------------------------
# --- Max-pooling 2x2 ----------
# -----------------------------
def maxpool2x2(X):
    # X: (N,F,H,W)
    N,F,H,W = X.shape
    X_reshaped = X.reshape(N,F,H//2,2,W//2,2)  # split into 2x2 blocks
    out = X_reshaped.max(axis=(3,5))           # (N,F,H//2,W//2)

    # broadcast properly: expand out along axes 3 and 5 to match X_reshaped
    mask = X_reshaped == out[:,:, :, None, :, None]  # shape: (N,F,H//2,2,W//2,2)
    return out, mask

def maxpool2x2_back(d_out, mask):
    # d_out: (N,F,H_out,W_out)
    N,F,H_out,W_out = d_out.shape
    dX = np.zeros((N,F,H_out*2,W_out*2), dtype=np.float32)
    # expand d_out to match mask shape
    d_out_expanded = np.repeat(np.repeat(d_out,2,axis=2),2,axis=3)
    # flatten for boolean assignment
    dX_flat = dX.reshape(-1)
    mask_flat = mask.reshape(-1)
    d_out_flat = d_out_expanded.reshape(-1)
    dX_flat[mask_flat] = d_out_flat[mask_flat]
    dX = dX_flat.reshape(N,F,H_out*2,W_out*2)
    return dX

# -----------------------------
# --- Network architecture ---
# -----------------------------
conv_filters = 8
filter_size = 3
fc_hidden = 128
output_size = 10

np.random.seed(42)
conv_w = np.random.randn(conv_filters, filter_size*filter_size*1) * np.sqrt(2/(filter_size*filter_size*1))
fc_w = np.random.randn(conv_filters*13*13, fc_hidden) * np.sqrt(2/(conv_filters*13*13))
fc_out_w = np.random.randn(fc_hidden, output_size) * np.sqrt(2/fc_hidden)

# -----------------------------
# --- Optimizers -------------
# -----------------------------
class Adam:
    def __init__(self, shape, lr=0.001):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.lr = lr
        self.beta1=0.9
        self.beta2=0.999
        self.eps=1e-8
        self.t=0
    def update(self, w, grad):
        self.t += 1
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        m_hat = self.m/(1-self.beta1**self.t)
        v_hat = self.v/(1-self.beta2**self.t)
        return w - self.lr*m_hat/(np.sqrt(v_hat)+self.eps)

class SGDMomentum:
    def __init__(self, shape, lr=0.001, momentum=0.9):
        self.v = np.zeros(shape, dtype=np.float32)
        self.lr = lr
        self.momentum = momentum
    def update(self, w, grad):
        self.v = self.momentum*self.v - self.lr*grad
        return w + self.v

adam_conv = Adam(conv_w.shape)
adam_fc = Adam(fc_w.shape)
adam_out = Adam(fc_out_w.shape)

sgd_conv = SGDMomentum(conv_w.shape, lr=0.001)
sgd_fc = SGDMomentum(fc_w.shape, lr=0.001)
sgd_out = SGDMomentum(fc_out_w.shape, lr=0.001)

# -----------------------------
# --- Training parameters -----
# -----------------------------
epochs = 5
batch_size = 128
num_samples = X_train.shape[0]

# -----------------------------
# --- Training loop ----------
# -----------------------------
for epoch in range(epochs):
    indices = np.random.permutation(num_samples)
    for start in range(0, num_samples, batch_size):
        end = start+batch_size
        X_batch = X_train[indices[start:end]]
        y_batch = y_train_oh[indices[start:end]]

        # --- Forward pass ---
        cols = im2col(X_batch, filter_size, filter_size)  # (N*out_h*out_w, kh*kw*C)
        conv_out = cols.dot(conv_w.T)  # (N*out_h*out_w, n_filters)
        N = X_batch.shape[0]
        out_h = out_w = 28 - filter_size + 1
        conv_out = conv_out.reshape(N, out_h, out_w, conv_filters).transpose(0,3,1,2)  # (N,F,H,W)

        pool_out, pool_mask = maxpool2x2(conv_out)
        flat = pool_out.reshape(N, -1)
        hidden = relu(flat.dot(fc_w))
        out = softmax(hidden.dot(fc_out_w))

        # --- Backprop ---
        d_out = (out - y_batch)/batch_size
        d_fc_out_w = hidden.T.dot(d_out)
        d_hidden = d_out.dot(fc_out_w.T) * relu_deriv(hidden)
        d_fc_w = flat.T.dot(d_hidden)
        d_flat = d_hidden.dot(fc_w.T).reshape(pool_out.shape)

        d_conv_out = maxpool2x2_back(d_flat, pool_mask)
        d_conv_out_cols = d_conv_out.transpose(0,2,3,1).reshape(-1,conv_filters)
        d_conv_w = d_conv_out_cols.T.dot(cols)

        # --- Update weights with dual optimizer ---
        conv_w = adam_conv.update(conv_w, d_conv_w)
        conv_w = sgd_conv.update(conv_w, d_conv_w)
        fc_w = adam_fc.update(fc_w, d_fc_w)
        fc_w = sgd_fc.update(fc_w, d_fc_w)
        fc_out_w = adam_out.update(fc_out_w, d_fc_out_w)
        fc_out_w = sgd_out.update(fc_out_w, d_fc_out_w)

    # Epoch loss
    loss = -np.mean(np.sum(y_batch*np.log(out+1e-8),axis=1))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# -----------------------------
# --- Test first 10 images ---
# -----------------------------
#X_test_small = X_test[:10]
#cols = im2col(X_test_small, filter_size, filter_size)
#conv_out = cols.dot(conv_w.T)
#out_h = out_w = 28 - filter_size + 1
#conv_out = conv_out.reshape(10, out_h, out_w, conv_filters).transpose(0,3,1,2)
#pool_out,_ = maxpool2x2(conv_out)
#flat = pool_out.reshape(10,-1)
#hidden = relu(flat.dot(fc_w))
#out = softmax(hidden.dot(fc_out_w))
#pred_labels = np.argmax(out, axis=1)

#print("Predictions for first 10 test images:", pred_labels)
#print("Actual labels:", y_test[:10])

# -----------------------------
# --- Full MNIST test evaluation ---
# -----------------------------
log_file = "mnist_test_log.txt"
batch_size_test = 256
num_test = X_test.shape[0]
correct = 0

with open(log_file, "w") as f:
    for start in range(0, num_test, batch_size_test):
        end = start + batch_size_test
        X_batch = X_test[start:end]
        y_batch = y_test[start:end]

        # --- Forward pass using your trained weights ---
        cols = im2col(X_batch, filter_size, filter_size)
        conv_out = cols.dot(conv_w.T)
        N = X_batch.shape[0]
        out_h = out_w = 28 - filter_size + 1
        conv_out = conv_out.reshape(N, out_h, out_w, conv_filters).transpose(0,3,1,2)
        pool_out,_ = maxpool2x2(conv_out)
        flat = pool_out.reshape(N, -1)
        hidden = relu(flat.dot(fc_w))
        out = softmax(hidden.dot(fc_out_w))
        pred_labels = np.argmax(out, axis=1)

        # --- Accuracy count ---
        correct += np.sum(pred_labels == y_batch)

        # --- Log predictions ---
        for p,a in zip(pred_labels, y_batch):
            f.write(f"{p} {a}\n")

accuracy = correct / num_test
print(f"Full MNIST test accuracy: {accuracy*100:.2f}%")
print(f"Predictions vs actual labels written to {log_file}")
