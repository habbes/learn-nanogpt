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

And the output block using the tensor:

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
