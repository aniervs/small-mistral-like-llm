# small-mistral-like-llm

Custom implementation of a small LLM with similar architectural details to Mistral7B:
- Grouped Query Attention
- Sliding Window Attention
- RMSNorm instead of LayerNorm
- SiLU in the feed-forward layer

## Details

This implementation was done by using `einops` and `pytorch`.
This was also an exercise for me to practice einops in a real-world scenario.

## TODO

Test it's ok in a different device from CPU (CUDA or MPS)
