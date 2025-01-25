# SmolLM2-135M Implementation

This is a PyTorch implementation of the SmolLM2-135M model, a smaller variant of LLaMA-style architecture.

## Model Architecture

- Parameters: ~135M
- Hidden Size: 576
- Intermediate Size (FFN): 1536
- Number of Layers: 30
- Attention Heads: 9
- KV Heads: 3 (grouped-query attention)
- Max Sequence Length: 2048
- Vocabulary Size: 49,152

### Parameter Calculation

Total parameters ≈ 135M, broken down as:

1. Token Embeddings: 49,152 × 576 = 28,311,552
2. Per Layer:
   - Self-attention:
     - Q projection: 576 × 576 = 331,776
     - K projection: 576 × 576 = 331,776
     - V projection: 576 × 576 = 331,776
     - O projection: 576 × 576 = 331,776
   - MLP:
     - Gate projection: 576 × 1536 = 884,736
     - Up projection: 576 × 1536 = 884,736
     - Down projection: 1536 × 576 = 884,736
   - Layer Norms: 576 × 2 = 1,152
   Total per layer: 3,482,464

3. Final Layer Norm: 576

Total = Embeddings + (Layers × Per Layer) + Final Layer Norm
     ≈ 28.3M + (30 × 3.48M) + 576
     ≈ 135M

## Training

The model is trained in two phases:
1. Initial training for 5000 steps with text generation every 500 steps
2. Additional 50 steps of fine-tuning

### Features
- Mixed precision training (bfloat16)
- Gradient accumulation
- Checkpoint saving and loading
- Text generation during training
- TensorBoard logging

## HuggingFace Spaces Demo

Try the model here: [HuggingFace Spaces Link]

## License

MIT