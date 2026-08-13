"""Transfer Whisper encoder weights to K3 audio encoder."""
import torch
import click
from .adapters import rescale_weights


def transfer_audio_encoder(whisper_model, k3_model, cfg):
    """Transfer Whisper encoder weights to K3AudioEncoder."""
    if k3_model.audio is None:
        click.secho("    ⚠ K3 has no audio encoder, skipping", fg="yellow")
        return 0

    click.echo("  📦 Transferring Whisper encoder → K3 audio...")

    transferred = 0
    encoder = whisper_model.model.encoder
    k3_audio = k3_model.audio

    # 1. Transfer conv layers (Whisper has 2 conv1d, K3 has WhisperStyleConv)
    if hasattr(encoder, 'conv1') and hasattr(encoder, 'conv2'):
        whisper_hidden = encoder.conv1.out_channels
        k3_hidden = cfg.audio_hidden

        click.echo(f"    📐 Conv layers: Whisper {whisper_hidden}d → K3 {k3_hidden}d")

        # Conv1: adapt output channels
        w1 = encoder.conv1.weight.data  # [whisper_hidden, 80, 3]
        if whisper_hidden >= k3_hidden:
            k3_audio.conv.conv1.weight.data.copy_(w1[:k3_hidden, :, :])
            if encoder.conv1.bias is not None:
                k3_audio.conv.conv1.bias.data.copy_(encoder.conv1.bias.data[:k3_hidden])
        else:
            k3_audio.conv.conv1.weight.data[:whisper_hidden].copy_(w1)
            if encoder.conv1.bias is not None:
                k3_audio.conv.conv1.bias.data[:whisper_hidden].copy_(encoder.conv1.bias.data)
        transferred += 1

        # Conv2: adapt both input and output channels
        w2 = encoder.conv2.weight.data  # [whisper_hidden, whisper_hidden, 3]
        if whisper_hidden >= k3_hidden:
            k3_audio.conv.conv2.weight.data.copy_(w2[:k3_hidden, :k3_hidden, :])
            if encoder.conv2.bias is not None:
                k3_audio.conv.conv2.bias.data.copy_(encoder.conv2.bias.data[:k3_hidden])
        else:
            k3_audio.conv.conv2.weight.data[:whisper_hidden, :whisper_hidden].copy_(w2)
            if encoder.conv2.bias is not None:
                k3_audio.conv.conv2.bias.data[:whisper_hidden].copy_(encoder.conv2.bias.data)
        transferred += 1

        click.secho("    ✓ Conv layers transferred", fg="green")

    # 2. Transfer positional embeddings (Whisper uses learned pos embed)
    if hasattr(encoder, 'embed_positions'):
        whisper_pos = encoder.embed_positions.weight.data  # [1500, whisper_hidden]
        k3_max_frames = cfg.audio_max_frames

        # Interpolate time dimension, adapt hidden dimension
        if whisper_pos.shape[0] != k3_max_frames:
            whisper_pos = torch.nn.functional.interpolate(
                whisper_pos.T.unsqueeze(0), size=k3_max_frames, mode='linear', align_corners=False
            ).squeeze(0).T

        if whisper_hidden >= k3_hidden:
            k3_audio.pos_embed.data.copy_(whisper_pos[:, :k3_hidden].unsqueeze(0))
        else:
            k3_audio.pos_embed.data[:, :, :whisper_hidden].copy_(whisper_pos.unsqueeze(0))

        transferred += 1
        click.secho("    ✓ Positional embeddings transferred", fg="green")

    # 3. Transfer transformer layers
    whisper_layers = encoder.blocks
    k3_layers = k3_audio.layers
    num_layers = min(len(whisper_layers), len(k3_layers))

    click.echo(f"    📦 Transferring {num_layers} transformer layers...")

    for i in range(num_layers):
        w_layer = whisper_layers[i]
        k3_layer = k3_layers[i]

        # Attention (Whisper uses MultiheadAttention, K3 uses same)
        if hasattr(w_layer, 'attn'):
            w_attn = w_layer.attn
            k3_attn = k3_layer.attn

            # Adapt Q, K, V projections
            for param_name in ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']:
                if hasattr(w_attn, param_name.split('.')[0]):
                    w_param = w_attn
                    k3_param = k3_attn
                    for part in param_name.split('.'):
                        w_param = getattr(w_param, part)
                        k3_param = getattr(k3_param, part)

                    # Adapt dimensions
                    if 'weight' in param_name:
                        if 'in_proj' in param_name:
                            # [3*hidden, hidden] for Q,K,V
                            w_data = w_param.data
                            if whisper_hidden >= k3_hidden:
                                # Truncate
                                adapted = w_data[:3*k3_hidden, :k3_hidden]
                            else:
                                # Pad
                                adapted = torch.zeros(3*k3_hidden, k3_hidden)
                                adapted[:3*whisper_hidden, :whisper_hidden] = w_data
                            k3_param.data.copy_(rescale_weights(adapted, target_std=0.02))
                        else:
                            # out_proj: [hidden, hidden]
                            if whisper_hidden >= k3_hidden:
                                k3_param.data.copy_(rescale_weights(w_param.data[:k3_hidden, :k3_hidden], target_std=0.02))
                            else:
                                k3_param.data[:whisper_hidden, :whisper_hidden].copy_(rescale_weights(w_param.data, target_std=0.02))
                    else:
                        # Bias
                        if 'in_proj' in param_name:
                            if whisper_hidden >= k3_hidden:
                                k3_param.data.copy_(w_param.data[:3*k3_hidden])
                            else:
                                k3_param.data[:3*whisper_hidden].copy_(w_param.data)
                        else:
                            if whisper_hidden >= k3_hidden:
                                k3_param.data.copy_(w_param.data[:k3_hidden])
                            else:
                                k3_param.data[:whisper_hidden].copy_(w_param.data)

        # FFN (Whisper uses fc1, fc2)
        if hasattr(w_layer, 'mlp'):
            w_mlp = w_layer.mlp
            k3_ffn = k3_layer.ffn

            # fc1: [hidden, hidden*4]
            if hasattr(w_mlp, 'fc1'):
                w_fc1 = w_mlp.fc1.weight.data  # [4*whisper_hidden, whisper_hidden]
                if whisper_hidden >= k3_hidden:
                    adapted = w_fc1[:4*k3_hidden, :k3_hidden]
                else:
                    adapted = torch.zeros(4*k3_hidden, k3_hidden)
                    adapted[:4*whisper_hidden, :whisper_hidden] = w_fc1
                k3_ffn[0].weight.data.copy_(rescale_weights(adapted, target_std=0.02))

                if hasattr(w_mlp.fc1, 'bias'):
                    if whisper_hidden >= k3_hidden:
                        k3_ffn[0].bias.data.copy_(w_mlp.fc1.bias.data[:4*k3_hidden])
                    else:
                        k3_ffn[0].bias.data[:4*whisper_hidden].copy_(w_mlp.fc1.bias.data)

            # fc2: [hidden*4, hidden]
            if hasattr(w_mlp, 'fc2'):
                w_fc2 = w_mlp.fc2.weight.data  # [whisper_hidden, 4*whisper_hidden]
                if whisper_hidden >= k3_hidden:
                    adapted = w_fc2[:k3_hidden, :4*k3_hidden]
                else:
                    adapted = torch.zeros(k3_hidden, 4*k3_hidden)
                    adapted[:whisper_hidden, :4*whisper_hidden] = w_fc2
                k3_ffn[2].weight.data.copy_(rescale_weights(adapted, target_std=0.02))

                if hasattr(w_mlp.fc2, 'bias'):
                    if whisper_hidden >= k3_hidden:
                        k3_ffn[2].bias.data.copy_(w_mlp.fc2.bias.data[:k3_hidden])
                    else:
                        k3_ffn[2].bias.data[:whisper_hidden].copy_(w_mlp.fc2.bias.data)

        # Layer norms (Whisper uses LayerNorm, K3 uses RMSNorm - only transfer scale)
        if hasattr(w_layer, 'self_attn_layer_norm'):
            w_norm = w_layer.self_attn_layer_norm.weight.data
            if whisper_hidden >= k3_hidden:
                k3_layer.attn_norm.weight.data.copy_(w_norm[:k3_hidden])
            else:
                k3_layer.attn_norm.weight.data[:whisper_hidden].copy_(w_norm)

        if hasattr(w_layer, 'final_layer_norm'):
            w_norm = w_layer.final_layer_norm.weight.data
            if whisper_hidden >= k3_hidden:
                k3_layer.ffn_norm.weight.data.copy_(w_norm[:k3_hidden])
            else:
                k3_layer.ffn_norm.weight.data[:whisper_hidden].copy_(w_norm)

        transferred += 1

    click.secho(f"    ✓ {num_layers} transformer layers transferred", fg="green")

    # 4. Transfer final layer norm
    if hasattr(encoder, 'layer_norm'):
        w_norm = encoder.layer_norm.weight.data
        if whisper_hidden >= k3_hidden:
            k3_audio.final_norm.weight.data.copy_(w_norm[:k3_hidden])
        else:
            k3_audio.final_norm.weight.data[:whisper_hidden].copy_(w_norm)
        transferred += 1
        click.secho("    ✓ Final layer norm transferred", fg="green")

    click.secho(f"    ✓ Audio encoder: {transferred} components transferred", fg="green")
    return transferred
