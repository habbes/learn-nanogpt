import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # How many independent sequences will we process in parallel?
block_size = 8  # what is the maximum content length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
eval_iters = 200
# --------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # First 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Data loading


def get_batch(split):
    # generate a small batch of data inputs x and target y
    data = train_data if split == "train" else val_data
    # generates a batch_size-sized sequence of random indexes (index ranginge from 0 to N - block_size)
    # Each index is an offset in the data to the start of a batch of size block_size.
    # So the last possible index would be N - block_size - 1 (the -1 so that we can have on item after the context block as expected output)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # The generated indices allow us to select a batch of random training blocks from the dataset
    # create a batch of training context blocks
    # torch.stack creates rows of data where each row corresponds to one of the input lists
    x = torch.stack([data[i : i + block_size] for i in ix])
    # create a batch of training predictions
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    # Set model to evaluation mode
    # For the simple bigram model, it doesn't make a difference.
    # But some layers like Dropout behave differently in train and eval mode.
    # So it's good practice to be explicit about which mode the model is in
    model.eval()
    for split in ["train", "val"]:
        # compute losse of a number of batches
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        # Get overage loss
        out[split] = losses.mean()
    
    # Set model to training mode
    model.train()
    return out


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # this is a lookup table that has vocab_size entries/rows and each
        # entry is a vector of size vocab_size
        # It's going to store predicted scores for the next character in the sequence
        # The lookup Embedding table is initialized with som random values, not with 0s
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers (batch of context blocks and corresponding batch of targets)
        # remember each encoded token in the input data as index into the vocab table, and therefore an index into
        # the embedding table as well
        # For each input integer in the (B,T) input batch, find its corresponding vector from the embedding table
        # So for row in B and col in T, we'll have a vector of size vocab_size (C)
        # So for all the input tokens, the lookup will return a tensor of size (B, T, C) where each input token is mapped to its embedding vector
        # The embedding vector for a given character represents the score of possible next character given the current character
        logits = self.token_embedding_table(
            idx
        )  # (B,T,C) C refers to channel, vocab size in this case

        if targets is None:
            loss = None
        else:
            # We use negative loss-likelihood or cross-entropy to compute the loss.
            # The cross_entropy function in pytorch expects the channels to be the second
            # dimension, so we need to reshape our data
            B, T, C = logits.shape
            # The B * T batches are linearized into a single vector where each element is an input token
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # generates the specified number of new tokens for each batch row
        # and append them to the initial tokens
        # idx is (B, T) array of indices in the current context

        # For the simple bigram model, we only use the last token to predict the next, so this
        # method is an overkill since it computes the logits for all the time steps. But it's written
        # to be generalizable and reusable for models with longer context windows
        for _ in range(max_new_tokens):
            # get the predictions
            # We use the model instance as a function to compute the output instead of calling self.forward(idx) directly. This is the recommended way according to the docs.
            # logits is a (B, T, C) array that maps each index in the current context to a vector of logits

            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[
                :, -1, :
            ]  # becomes (B, C), i.e, in each row of the batch, take only the last col
            # apply softmax to get probabilities along the C dimension (i.e. the logits), such that for each embedding vector, its values are scaled to [0, 1] and sum up to 1
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            # for each row in the batch, select the next character index based on the probabilities
            # TODO: Why multinomial distribution?
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sample index to the running sequence so it will be used in the prediction of the next token
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx

model = BigramLanguageModel(vocab_size)
# Move the model (i.e. its parameters) to the device
m = model.to(device)

# create PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on trian and val sets
    if iter % eval_interval == 0:
        # Instead of printing loss for every batch (which could be noisy)
        # estimate loss over multiple batches
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))