# Attention Mask

- Why is attention mask lower triangular matrix `torch.tril`?
	- Attention score matrix $A \in R^{T \times T}$ where T = seq_len
		- $A=QK^T$, $A[i,j]=<q_i, k_j>$
		- row i = query position (token currently “looking”)
		- column j = key position (token being “looked at”)
		- Thus the attention mask M[i, j] signifies whether query i should look at key position j. 
		- Causal mask M[i,j]={1, ​if j<=i and 0, otherwise​



- Attention mask typically applies only in Prefill stage.
	- In prefill, the attention matrix is (T x T)
	- In decode, the attention matrix is (1 x T\_prev). Query is length 1 (the current token), Keys are length T\_prev (past tokens).


## Prefill Stage

```python
import torch
import torch.nn.functional as F
import math

B, H, T, D = 1, 1, 5, 4

q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)
v = torch.randn(B, H, T, D)

scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
print("raw scores:\n", scores[0,0])

mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~mask, float("-inf"))
probs = F.softmax(scores, dim=-1)

print("\nmasked scores:\n", scores[0,0])
print("\nprobs:\n", probs[0,0])

```
This sets top right corner to -inf (so that softmax gives zero to masked positions), and bottom right corner is normal numbers. 


## Decode Stage
```python
# decoding token t=4

t = 4  # 0-based

q = torch.randn(B, H, 1, D)      # query for token 4
k_all = torch.randn(B, H, t, D)  # stored keys 0..3
v_all = torch.randn(B, H, t, D)

scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(D)
probs = F.softmax(scores, dim=-1)

print("\ndecode scores:\n", scores[0,0])
print("\ndecode probs:\n", probs[0,0])

```

In decode stage, causal mask not required.