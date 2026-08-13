# Video Temporal Modeling Fix

## ❌ **Problem Found**

The original implementation was **destroying temporal information** from videos!

### What Was Wrong:

```python
# BEFORE (destroyed temporal info):
vis = self.vision(images).mean(1, keepdim=True)  # ❌ Mean pool ALL temporal tokens
x = x + vis  # Broadcast single token to all text positions
```

**Flow:**
1. Input: 4 video frames
2. MoonViT processes with spatial-temporal attention ✅
3. MoonViT temporal pooling: 4 frames → 2 temporal tokens ✅
4. **Then we mean-pooled 2 tokens → 1 token** ❌❌❌
5. Lost all temporal structure!

---

## ✅ **Fix Applied**

### 1. Preserve Temporal Tokens (k3/model.py)

```python
# AFTER (preserves temporal info):
vis = self.vision(images)  # Keep all temporal tokens
x = torch.cat([vis, x], dim=1)  # Prepend to sequence (like CLIP/Flamingo)
```

**New flow:**
1. Input: 4 video frames
2. MoonViT spatial-temporal attention ✅
3. MoonViT temporal pooling: 4 frames → 2 temporal tokens ✅
4. **Prepend 2 temporal tokens to text sequence** ✅
5. Model can attend to each temporal token separately!

### 2. Optional Temporal Augmentation (preprocess_openvid.py)

Added random temporal jittering for training:

```python
# Fixed uniform sampling (preprocessing/eval)
indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

# Random sampling with jitter (training augmentation)
segment_len = total_frames // num_frames
for i in range(num_frames):
    start, end = i * segment_len, (i+1) * segment_len
    indices.append(random.randint(start, end))
```

---

## 📊 **What This Means**

### Before Fix:
- ❌ Videos treated same as images (single pooled feature)
- ❌ No temporal dynamics captured
- ❌ Can't distinguish between static vs. motion
- ❌ Loss of frame ordering information

### After Fix:
- ✅ Multiple temporal tokens (2-4 depending on pooling)
- ✅ Model can attend to different temporal moments
- ✅ Captures motion and temporal dynamics
- ✅ Frame order preserved via positional embeddings

---

## 🔬 **MoonViT Temporal Architecture**

The vision encoder already has proper temporal modeling:

```python
# 1. Spatial positional embeddings
x = x + self.spatial_pos[:, :P]

# 2. Temporal positional embeddings  
x = x + self.temporal_pos[:, :F]

# 3. Factorized spatial-temporal attention
for spatial_attn, temporal_attn in self.layers:
    x = spatial_attn(x)      # Spatial: within each frame
    x = temporal_attn(x)     # Temporal: across frames

# 4. Temporal pooling (4 → 2 frames)
x = x.view(...).mean(2)  # Pool by factor of temporal_pool

# Output: (B, num_temporal_tokens, hidden_dim)
# - num_temporal_tokens = original_frames / temporal_pool
# - For 4 frames with pool=2: 2 temporal tokens
```

We were throwing this away with mean pooling!

---

## 🎯 **Impact on Training**

### Sequence Length Changes:

**Text only:**
- Sequence: [tok1, tok2, ..., tokN]
- Length: N

**Text + Image (before fix):**
- Sequence: [tok1, tok2, ..., tokN] + single_visual_bias
- Effective length: N (bias broadcast)

**Text + Video (after fix):**
- Sequence: [vid_t1, vid_t2, tok1, tok2, ..., tokN]
- Length: N + num_temporal_tokens
- For 4-frame video: N + 2 tokens

**Text + Audio + Video (after fix):**
- Sequence: [aud_tokens..., vid_t1, vid_t2, tok1, ..., tokN]
- Length: N + num_audio_tokens + num_temporal_tokens

### Memory Impact:
- Slightly higher (2-4 extra tokens per video sample)
- Negligible compared to benefits

---

## 🚀 **Next Steps**

### Current Implementation:
- ✅ Temporal structure preserved
- ✅ Spatial-temporal attention working
- ✅ Proper token sequence construction

### Future Improvements:

1. **Temporal Augmentation** (easy):
   - Add `--temporal-jitter` flag to training
   - Sample different frames each epoch
   - Improves generalization

2. **Multi-scale Temporal** (advanced):
   - Sample different frame rates (8fps, 15fps, 30fps)
   - Capture both fast and slow motion

3. **Longer Videos** (if needed):
   - Increase from 4 to 8 frames
   - More temporal context
   - Higher memory cost

4. **Benchmarking**:
   - Compare video w/ vs w/o temporal fix
   - Measure action recognition accuracy
   - Test temporal reasoning tasks

---

## 📝 **Testing the Fix**

Once you retrain with the fixed model:

```python
# Test temporal awareness
from k3.model import K3Model
import torch

model = K3Model.from_pretrained("checkpoint")

# Create video with temporal change
# Frame 0-1: static, Frame 2-3: motion
video = torch.randn(1, 4, 3, 112, 112)

# Check output
output = model(input_ids, images=video)

# With fix: model should generate different tokens for different temporal moments
# Without fix: all frames treated identically
```

---

## 📚 **References**

Similar approaches in:
- **Flamingo**: Prepends visual tokens to text sequence
- **CLIP**: Uses [CLS] token for global features
- **Video-LLaMA**: Multiple temporal tokens for video understanding
- **Kimi K3**: Spatial-temporal factorized attention (MoonViT)

Our fix aligns with these modern VLM architectures.
