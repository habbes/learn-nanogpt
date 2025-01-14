import torch
import torch.nn as nn
from torch.nn import functional as F

# This version adds a simple linear layer to the bigram implementation

# hyperparameters
batch_size = 32  # How many independent sequences will we process in parallel?
block_size = 8  # what is the maximum content length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3 # user lower learning rate cause the self-attention doesn't tolerate very high learning rates
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
eval_iters = 200
n_embed = 32 # size of embedding vector for each item in our vocab
n_head = 4 # number of heads in the multi-head attention
n_layers = 6 # number of self-attention->feedforward blocks in the model
dropout = 0.2 # dropout rate for regularization, to prevent overfitting (randomly drops out some neurons during training). See: https://dl.acm.org/doi/pdf/10.5555/2627435.2670313
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

class SelfAttentionHead(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        # we create key, query, and value linear layers to compute
        # the key, query and value vectors for each token in the input sequence
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # tril is not a parameter of the module, we created it using the register_buffer
        # method so that it won't be updated by the optmizer
        # the tril is a lower triangular matrix of ones
        # used to mask away the future tokens in the self-attention mechanism
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Added dropout when scaling to avoid overfitting?
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        # apply the key, query layers to the input
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # compute attention scores ('affinities') between each pair of tokens
        # the attention scores are computed as the dot product between the query and key vectors
        # the 1/sqrt(C) scaling factor is used to prevent the dot products from getting too large or too small
        # the keeps the variance around 1, making them suitable for the softmax function
        wei = q @ k.transpose(-2, -1) * C ** (-0.5) # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        # mask away the future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        wei = self.dropout(wei)

        # apply the attention scores to the value vectors to get the weighted sum of values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # Add projection layer that will go back into the residual pathway. Why?
        # Added dropout when scaling to avoid overfitting?
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # apply each head to the input in parallel
        # concatenate them over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, head_size * num_heads)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # in the paper, the feedfoward layer is 4 times the size of the embedding dimension
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed) # add a projection layer that will go back into the residual pathway. Why?
            # Add dropout before the residual connection (This was done to help scale the neural net, avoid overfitting?)
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block that encapsulates communication followed by computation,
    i.e. self-attention followed by feedforward layer"""

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        # self-attention head modules
        # instead of one large self-attention head, we use multiple smaller heads in parallel
        # this improved the accuracy of the model
        # it helps to have multiple heads to learn different aspects of the data
        self.sa = MultiHeadAttention(n_head, head_size)
        # add a simple feedforward layer to the model to give the nodes to process what they've learned
        # from each other before computing the logits or the next step
        self.ffwd = FeedForward(n_embed)
        # The paper uses layer normalization, this helps reduce the training time
        # layer norm is like batch norm, but normalizes rows instead of columns
        # layer norm is described in this paper https://arxiv.org/pdf/1607.06450
        # In the video, he explains it more using the BatchNorm implementation
        # from his previous series: https://youtu.be/kCc8FmEb1nY?feature=shared&t=5571
        # We'll apply layer norm before the self-attention and feedforward layers
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        # Since we're now adding multiple blocks into the overal model,
        # the network is getting deeper and harder to optimize.
        # One trick to help with optimization of deeper networks is to use residual connections,
        # where we add the input to the output of the block before applying the non-linearity.
        # i.e. the input is moved to the next step, but is also "forked" separately into some
        # computation that is added back to the input before the next step.
        # See "Deep Residual Learnin for Image Recognition, 2015": https://arxiv.org/pdf/1512.03385
        # And "Understanding ResNet architecture": https://medium.com/@ibtedaazeem/understanding-resnet-architecture-a-deep-dive-into-residual-neural-network-2c792e6537a9
        # Addition is used because it distributes the gradients evenly to both paths during back progagation.

        # apply layer normalization before the self-attention head
        # this is a deviation from the original paper, which applied layer norm after,
        # but it's common practice today. It's called pre-norm.
        x = x + self.sa(self.ln1(x))
         # the feedforward layer processes the output of the self-attention head
        # on a token-by-token basis. All the tokens do this independently.
        # The self-attention is the communication to gather the data, then now the tokens have to "think"
        # about that data individually.
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # The table is size (vocab_size, n_embed)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # it is common to not only encode the "identities" of the token, but also the position
        # each position will get its own embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # chain multiple blocks of self-attention and feedforward layers
        # to intersperse communication and computation
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layers)])
        
        # There's also usually a layer norm after the last block,
        # but before the final linear layer that feeds into the vocabulary
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        # linear layer to convert embeddings into logits, i.e. likelihoods of each character in the vocab to be the next character
        self.lm_head = nn.Linear(n_embed, vocab_size) # lm -> language model

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets ar eboth (B, T) tensor of integers
        token_embeddings = self.token_embedding_table(
            idx
        )  # (B,T,C) C refers to channel, n_embed in this case

        # torch.arange returns a sequence of integers, i.e. indexes, from 0 to T-1
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        # Combine token embeddings and position embeddings
        # X holds not only the token identities, but also the positions at which these tokens occur
        # This doesn't add much info for now since our model does not consider
        # history other than the last token, but it will be relevant
        # when we move forward and talk about attention
        x = token_embeddings + position_embeddings # (B, T, C)
        # feed the embeddings through the self-attention head
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

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

            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
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

model = BigramLanguageModel()
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