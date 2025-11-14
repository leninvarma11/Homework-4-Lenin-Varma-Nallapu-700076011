# Homework-4 -Explanation
# Lenin-Varma-Nallapu
# 700076011

# 1 Question

### **a) Why each task maps to a specific RNN I/O pattern**

* **Next-word prediction → Many-to-One**
  In next-word prediction, the model reads an entire sequence of previous words and produces **one prediction**: the next word. Even though generation happens repeatedly during inference, each prediction step takes a sequence as input and outputs a single token, which matches the many-to-one pattern.

* **Sentiment analysis → Many-to-One**
  Sentiment classification requires processing the **full sentence** to understand its emotional tone, but it outputs only **one label** (positive, negative, neutral). This naturally fits the many-to-one setup because multiple input tokens lead to a single decision.

* **Named Entity Recognition (NER) → Many-to-Many (aligned)**
  In NER, each word in the input sentence must get a corresponding tag (e.g., PER, ORG, LOC). The number of outputs matches the number of inputs, and they are aligned token-by-token. This is exactly what a **many-to-many aligned** RNN does—process a sequence and produce one output per timestep.

* **Machine Translation → Many-to-Many (unaligned)**
  In translation, the input and output sentences usually have **different lengths**. The encoder reads the full input sequence, and the decoder generates a new sequence with no one-to-one alignment between tokens. This fits **many-to-many unaligned**, where a separate encoder–decoder structure handles variable-length outputs.

### **b) How unrolling enables BPTT and weight sharing**

When an RNN is unrolled, the repeated recurrence across time is expanded into a series of connected layers—one layer per timestep. This allows gradients to be propagated through each timestep during **Backpropagation Through Time (BPTT)**, while still using the **same set of recurrent weights** at every step, preserving temporal weight sharing.

### **c) Advantage and limitation of weight sharing across time**

* **Advantage — fewer parameters & better generalization**
  Because the same weights are reused at every timestep, the model is much more parameter-efficient. This not only reduces memory and training cost but also helps the RNN generalize patterns (like grammar or rhythm) regardless of where they appear in the sequence.

* **Limitation — cannot learn position-specific rules**
  With shared weights, the model treats every timestep identically. This makes it harder for the RNN to learn behaviors that depend on absolute position (e.g., “the first word is usually a subject” or “the last word often completes the idea”), because it cannot specialize different weights for different positions.


# 2 Question

### **a) What is the vanishing gradient problem and why it hurts long-range learning?**

In recurrent neural networks (RNNs), gradients must be propagated backward through many time steps during training. As this backward signal moves through repeated multiplication with weights and activation derivatives, it often **shrinks exponentially**. When gradients become extremely small, the early timesteps receive almost no learning signal. As a result, the network fails to capture **long-range dependencies**, meaning it struggles to relate information from earlier in the sequence to later predictions.

### **b) Architectural solutions that improve gradient flow**

* **LSTM (Long Short-Term Memory):**
  LSTMs introduce special gates—forget, input, and output—along with a dedicated memory cell. The cell state provides a nearly linear path for gradients, allowing information to persist across many timesteps and preventing gradients from vanishing.

* **GRU (Gated Recurrent Unit):**
  GRUs simplify the LSTM design with update and reset gates. These gates control how much past information to keep or discard, enabling more efficient gradient flow and helping the model remember longer-term patterns without suffering from vanishing gradients as severely as vanilla RNNs.

### **c) Training technique to mitigate the issue**

* **Gradient clipping:**
  During backpropagation, gradient clipping restricts gradients to a predefined maximum range. This prevents extremely small or extremely large gradient values and stabilizes training. While primarily used to prevent exploding gradients, it also helps maintain healthier gradient magnitudes overall, improving the model’s ability to learn temporal dependencies.

# 3 Question

### **a) Roles of the forget, input, and output gates**

LSTMs introduce three gating mechanisms that regulate how information flows through time:

* **Forget Gate (sigmoid σ):**
  This gate decides **how much of the previous cell state should be retained or removed**. Its sigmoid activation outputs values between 0 and 1, where 0 means “completely forget” and 1 means “fully keep” the previous memory.

* **Input Gate (sigmoid σ + tanh):**
  The input gate determines **what new information should be written** into the cell state.

  * The **sigmoid** component chooses which parts of the candidate information to update.
  * The **tanh** component generates candidate values that can be added to the cell state.

* **Output Gate (sigmoid σ + tanh):**
  This gate controls **how much of the internal cell state should be revealed** as the hidden output at that timestep. The sigmoid decides the exposure level, while tanh squashes the cell state into a usable output range.

### **b) Why the LSTM cell state provides a “linear path” for gradients**

The LSTM cell state is updated through **element-wise additions and gated multiplications**, rather than through repeated non-linear transformations. This structure creates an almost **linear computational path**, which makes it easier for gradients to pass backward through many timesteps. As a result, LSTMs avoid the vanishing gradient problem and can learn long-term dependencies more effectively than standard RNNs.

### **c) “What to remember” vs. “what to expose” in LSTMs**

LSTMs separate internal memory storage from what the network outputs:

* **“What to remember”** is governed by the **forget gate** and **input gate**, which decide what information should stay inside the cell state over time.
* **“What to expose”** is handled by the **output gate**, which determines how much of that internal memory should become the visible hidden state at the current step.

# 4 Question

### **a) What are Query (Q), Key (K), and Value (V)?**

In self-attention, each token in a sequence is transformed into three different vectors:

* **Query (Q):**
  Represents what the current token wants to *search for* in other tokens. It defines the type of information the token is trying to retrieve.

* **Key (K):**
  Represents what information a token *offers*. Keys act like labels that determine how well each token matches a given query.

* **Value (V):**
  Contains the actual information or features that will be aggregated to produce the final contextual representation after attention weighting.

Self-attention works by comparing every Query with every Key to compute attention weights, which are then used to combine the Value vectors.

### **b) Formula for dot-product attention**

The standard scaled dot-product attention is defined as:

• Attention(Q, K, V) = softmax( (Q · Kᵀ) / √dₖ ) V 

This formula captures how strongly each query interacts with all keys, normalizes those interactions with a softmax, and then uses them to compute a weighted sum of the value vectors.

### **c) Why divide by √dₖ?**

As the dimensionality ( d_k ) increases, the raw dot products ( Q \cdot K ) naturally grow larger in magnitude. Large logits can push the softmax function into extremely peaked outputs or even numerical instability, where gradients become very small. Dividing by ( \sqrt{d_k} ) **normalizes the scale** of the dot product, keeping the logits in a stable range so softmax produces smoother probabilities and the model trains more effectively.

# 5 Question

### **a) Why Transformers use multi-head attention**

Transformers use multiple attention heads because each head can focus on **different types of relationships** within the sequence. By projecting the queries, keys, and values into different learned subspaces, each head can capture unique patterns—such as positional relationships, syntactic structure, or semantic meaning. This parallel attention mechanism allows the model to build a much richer and more expressive representation than what a single attention head could learn on its own.

### **b) Purpose of Add & Norm (Residual + LayerNorm)**

The Add & Norm block combines two important ideas that stabilize and strengthen training:

* **Residual connections:**
  These connections add the input of a layer back to its output, creating a shortcut path for gradients. This helps prevent vanishing gradients and enables the network to train effectively even at great depth.

* **Layer Normalization:**
  After the residual addition, LayerNorm normalizes the activations to maintain consistent scaling. This improves training stability, helps the model converge faster, and reduces sensitivity to initialization and learning rate.

Together, Add & Norm ensures smoother optimization and stronger gradient flow through the model.

### **c) Example of linguistic relations that different heads may capture**

Different attention heads often learn to specialize in distinct linguistic patterns. For example, one head may track **subject–verb agreement**, ensuring that phrases like “dogs run” and “dog runs” remain grammatically coherent. Another head might learn **coreference patterns**, linking a name such as “John” to pronouns like “he” within the same sentence.

# 6 Question

### **a) Why the decoder uses masked self-attention**

In the decoder, masked self-attention is used to **block access to future tokens** during training. This ensures that the model cannot “peek ahead” at words that it is supposed to predict. Without masking, the decoder could cheat by using future information, leading to **information leakage**. Masking enforces strict left-to-right generation so the model learns to predict each token only from previously generated tokens.

### **b) Difference between encoder self-attention and encoder–decoder cross-attention**

* **Encoder self-attention:**
  Each token in the input sequence attends to **all other tokens in the same sequence**, enabling the encoder to build rich contextual representations of the entire input sentence.

* **Encoder–decoder cross-attention:**
  During decoding, the decoder's query vectors attend to the **outputs of the encoder**. This allows the decoder to incorporate information from the encoded source sentence while generating output tokens, linking the input and output sequences together.

### **c) How the model generates tokens step-by-step during inference (no teacher forcing)**

During inference, the model must generate text **one token at a time**:

1. The decoder is first given a special start-of-sentence token (`<sos>`) to begin generation.
2. The decoder predicts the next token, and that predicted token is fed back as the next input.
3. Masked self-attention ensures it only uses previously generated tokens.
4. This loop continues — each predicted token becomes the next input — until the model outputs an end-of-sequence token (`<eos>`).

This iterative process allows the model to produce entire sentences or sequences during generation.



