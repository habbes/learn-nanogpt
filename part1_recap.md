# Part 1 Recap

Quick recap of the lecture up to timestamp [1:01:57](https://youtu.be/kCc8FmEb1nY?feature=shared&t=3717) to refresh what I've learned before I jump in to part 2.

## Data and encoding

Load training text and create a basic character-level encoder/decoder pair:

- The vocabulary is the set of all characters that appear in the training text. The total number of chars = `vocab_size`
- Each char in the vocabulary is assigned a unique index.
- `encode`: transforms input text by mapping chars to their indices. `decode` transforms tokens into text via reverse operation.

Encode the entire training set into a pytorch tensor, split into training set and validation set (90%/10% split in this example).

In our example, each character represents a single token. But production models have more sophisticated tokenization techniques.
Here are examples of "real-world" tokenizers:

- [Google's SentencePiece](https://github.com/google/sentencepiece)
- [OpenAI's tiktoken](https://github.com/openai/tiktoken)

## Context Length and Prediction

Here's the goal of the neural network we want to train: Given an sequence of tokens as input, predict the next token.

When training, we don't pass the entire data set to the transformer at once as it would be computationally prohibitive. We'll break the
data into chunks. Then we sample random chunks from the training set and traing chunks at a time. These chunks have
a maximum length, which we'll call the block size or **context length**.

Furthermore, each of block of tokens provides may training examples. Each subset `block[:t]` is training sample that predicts the token `block[t]`.

In this example we use a block size of 8.

For example, let the following sequence of tokens be a chunk from the training data. It contains 9 elements instead of 8 because the 9th element
is the target output of the block that contains all te first 8 items.

```python
[18, 47, 56, 57, 58, 1, 15, 47, 58]
```

We can generate the following training examples:

| context | target output |
|----------|---------------|
|`[18]`    | `47` |
|`[18, 47]` | `56` |
|`[18, 47, 56]` | `57` |
|`[18, 47, 56, 57]` | `58` |
|`[18, 47, 56, 57, 58, 1]` | `15` |
|`[18, 47, 56, 57, 58, 1, 15]` | `47` |
|`[18, 47, 56, 57, 58, 1, 15, 47]` | `58` |

We can represent the input block using the tensor:

```python
x = [18, 47, 56, 57, 58, 1, 15, 47]
```

And the corresponding target outputs using the tensor:

```python
y = [47, 56, 57, 58, 1, 15, 47, 58]
```

Notice that `y` is the block shifted one item to the right. At each index `t`, the subsequence of all elements in `x` up to `t` (inclusive) maps to the
`y` element at `t`, i.e: for each t in range `block_size`, `x[:t + 1]` maps to the target output `y[t]`.

Or more explicitly:

```python
# Let's the concept in the previous block into code
x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
  context = x[:t + 1]
  target = y[t]
  print(f"when input is {context} the target: {target}")
```

## Batch processing

Instead of training one block at a time, we traing a batch of blocks at the same time to take advantage of the GPU's parallelism.
A batch contain a set of sequences that will be processed in parallel. Blocks in the batch are independent from each other, i.e. they don't "talk" to each other.

In this example, we'll use a batch size of 4.

Each batch contains a stack of input tensors and a corresponding stack of output tensors.

Input tensor in the batch:

```python
xb = 
tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])
```

Output tensor in the batch:

```python
yb =
tensor([[43, 58,  5, 57,  1, 46, 43, 39],
        [53, 56,  1, 58, 46, 39, 58,  1],
        [58,  1, 58, 46, 39, 58,  1, 46],
        [17, 27, 10,  0, 21,  1, 54, 39]])
```

Each row in the input batch tensor has a corresponding target row in the output batch.

Here's sample code that demonstrates how a batch can be generated:

```python
def get_batch(data):
  # generate a small batch of data inputs x and target y
  # generates a batch_size-sized sequence of random indexes (index ranginge from 0 to N - block_size)
  # Each index is an offset in the data to the start of a batch of size block_size.
  # So the last possible index would be N - block_size - 1 (the -1 so that we can have on item after the context block as expected output)
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  # The generated indices allow us to select a batch of random training blocks from the dataset
  # create a batch of training context blocks
  # torch.stack creates rows of data where each row corresponds to one of the input lists
  x = torch.stack([data[i:i + block_size] for i in ix])
  # create a batch of training predictions
  y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
  return x, y

# get a random batch from training data
xb, yb = get_batch(training_data)
```

And here's a loop that demonstrates how the batch would be interpreted:

```python
for b in range(batch_size): # batch dimension
  for t in range(block_size): #time dimension
    context = xb[b, :t + 1]
    target = yb[b, t]
    print(f"when input context is {context.tolist()} the target is: { target}")
```

The output would look like

```text
when input context is [24] the target is: 43
when input context is [24, 43] the target is: 58
when input context is [24, 43, 58] the target is: 5
when input context is [24, 43, 58, 5] the target is: 57
when input context is [24, 43, 58, 5, 57] the target is: 1
when input context is [24, 43, 58, 5, 57, 1] the target is: 46
when input context is [24, 43, 58, 5, 57, 1, 46] the target is: 43
when input context is [24, 43, 58, 5, 57, 1, 46, 43] the target is: 39
when input context is [44] the target is: 53
when input context is [44, 53] the target is: 56
when input context is [44, 53, 56] the target is: 1
when input context is [44, 53, 56, 1] the target is: 58
when input context is [44, 53, 56, 1, 58] the target is: 46
when input context is [44, 53, 56, 1, 58, 46] the target is: 39
when input context is [44, 53, 56, 1, 58, 46, 39] the target is: 58
when input context is [44, 53, 56, 1, 58, 46, 39, 58] the target is: 1
when input context is [52] the target is: 58
when input context is [52, 58] the target is: 1
when input context is [52, 58, 1] the target is: 58
when input context is [52, 58, 1, 58] the target is: 46
when input context is [52, 58, 1, 58, 46] the target is: 39
when input context is [52, 58, 1, 58, 46, 39] the target is: 58
when input context is [52, 58, 1, 58, 46, 39, 58] the target is: 1
when input context is [52, 58, 1, 58, 46, 39, 58, 1] the target is: 46
when input context is [25] the target is: 17
when input context is [25, 17] the target is: 27
when input context is [25, 17, 27] the target is: 10
when input context is [25, 17, 27, 10] the target is: 0
when input context is [25, 17, 27, 10, 0] the target is: 21
when input context is [25, 17, 27, 10, 0, 21] the target is: 1
when input context is [25, 17, 27, 10, 0, 21, 1] the target is: 54
when input context is [25, 17, 27, 10, 0, 21, 1, 54] the target is: 39
```

## Simplest neural network for language models: BiGram language model

A bigram model predicts the next token solely based on the current token, it doesn't
take the history or sequence of previous tokens into account.

Our Bigram model is going to simple: Store a lookup table that maps each token in the vocabulary to a vector of scores or likelihoods for the next tokens and update these scores during training.

Let's call this table `token_embedding_table`. Let `C` be the vocabulary size.

The embedding table will be a C x C table. Each row of the table corresponds to the embedding vector for the token
at that index in the vocabulary.

i.e. `token_embedding_table[i, j]` returns the likelihood that the token corresponding to `vocab[j]` will follow the token at `vocab[i]` in the text.

In this simple model, these scores are actually unnormalized logits, like one would get from the cross_entropy loss function.
We initialize this table using the [`nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html).

During the forward pass of the neural network, we take the logits corresponding to the input batch, i.e. for each batch, for each block, for each token, get the embedding vector of logits corresponding to that token.
The result is a (B, T, C) matrix where B = batch size, T = block size, C = vocab size.

We'll use the [`F.cross_entropy`](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy) to compute the loss between the logits (which represents the predicted scores) and the actual output classes.
The output classes here correspond to the actual "next tokens" as retrieved from the target tensor.

Remember from the previous section that the target tensor is a B x T matrix where B is the batch size and T is the block size. Each token in `Y[b, t]` is the target output of the input sub-block `X[b, :t + 1]`. However, since this is a bigram model, that only considers the current token when predicting the next token, the input for each target `Y[b, t]` is actual `X[b, t]` rather than the sub-block `X[b, :t + 1]`.

Also remember, that tokens are actually indices already. So we can use them directly into the `cross_entropy` function.

The `cross_entropy` function expects a tensor `(N, C)` where `M` is the size of the minibatch. To be compatible with this format, we have to reshape our `(B, T, C)` tensor into `(B * T, C)` where the blocks are flattened. So each row of the input maps to the logits vector of the corresponding token. We do the same reshaping of the target tensor (from `(B , T)` to `(B * T)`) into a vector where each item is the target output token corresponding to the input token at that index.

The `cross_entropy` returns the [cross-entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) between the input and target.

The forward pass returns both the logits and the loss.

In order to generated predicted tokens from an input batch:

- we run the forward pass of the model to get the logits (scores) corresponding to the input batch.
- Then we take embedding vectors corresponding to the last token/column in each block of the batch (since this is a bigram model)
- use the [`softmax`](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) function to normalize the logits into probabilities
- use a sampling function, in this case [`multinomial`](https://pytorch.org/docs/stable/generated/torch.multinomial.html#torch-multinomial) to select a new token based on the computed probabilities
- append the predicted tokens to their corresponding blocks (one per batch item)
- repeat this process for as many tokens you want to generate (this ends up using generated tokens to predict more tokens)

