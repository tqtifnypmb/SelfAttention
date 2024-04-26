# This repo contains an example that demonstrates the issues I encountered in CoreML

### The peculiar behavior of matrix multiplication.

I converted a transformer model into a CoreML model and encountered a calculation error when computing attention with kv_cache. The specific behavior is as follows.

In matrix multiplication:

`query * kv_cache`

The shape of query is `[batch_size, seq_length, state_len]`

In mathematics, in the sequence dimension, each row is independent and does not affect each other. (kv_cache makes sense based on this principle). However, in CoreML, this independence does not hold true.

An example is in GPT-like models, theoretically, it only requires the previous token along with the kv_cache of all tokens to compute the next token. However, after converting the model to CoreML, following the theoretical approach would yield incorrect results. It is necessary to input all past tokens and recalculate in order to obtain the correct next token.

In other words, if 

`value_1 = [batch_size, 1, state_len] * kv_cache `

and 

`value_2 = [batch_size, 150, state_len] * kv_cache. `

Then 

`value_1[:, -1]` should be equal to `value_2[:, -1]` if the last row of data is the same in the matrix multiplication on the sequence dimension in the above two cases. However, this does not hold true in CoreML.

# Based on my past experience, I speculate that the possible cause of this problem is:

The CoreML is not suitable for intermediate calculations. 

If a computation is divided into two parts, with one part completed inside CoreML and the other part completed outside CoreML, the final result is likely to have significant errors. It is necessary to ensure that the entire calculation occurs within CoreML.
