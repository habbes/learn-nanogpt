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

For example, let the following sequence of tokens be a chunk from the training data:

```
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