import torch
import click
from transformers import AutoConfig, AutoModelForCausalLM
from k3.config import K3HFConfig
from k3.model import K3ForCausalLM
from k3.tokenizer import get_k3_tokenizer

AutoConfig.register("k3", K3HFConfig)
AutoModelForCausalLM.register(K3HFConfig, K3ForCausalLM)


@click.command()
@click.option("--checkpoint", type=str, default="checkpoints/k3_epoch2.pt", help="checkpoint to convert")
@click.option("--output", type=str, default="lv12_MicroK3", help="output directory")
def main(checkpoint, output):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    hf_config = K3HFConfig(
        vocab_size=cfg.vocab_size,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        layers_per_block=cfg.layers_per_block,
        num_heads=cfg.num_heads,
        use_vision=cfg.use_vision,
        use_mtp=cfg.use_mtp,
        vit_num_frames=cfg.vit_num_frames,
        vit_hidden=cfg.vit_hidden,
        vit_layers=cfg.vit_layers,
    )

    model = K3ForCausalLM(hf_config)
    model.model.load_state_dict(ckpt["model_state_dict"])

    model.config.tie_word_embeddings = False
    if model.model.mtp is not None:
        model.model.mtp.embed = torch.nn.Embedding.from_pretrained(model.model.embed.weight.clone())
    if model.model.lm_head.weight is model.model.embed.weight:
        model.model.lm_head.weight = torch.nn.Parameter(model.model.embed.weight.clone())

    model.save_pretrained(output, safe_serialization=False)
    hf_config.save_pretrained(output)

    tokenizer = get_k3_tokenizer()
    tokenizer.save_pretrained(output)

    print(f"✓ Saved to {output}/")
    print(f"  Vision: {cfg.use_vision}, MTP: {cfg.use_mtp}")
    print(f"\nPublish:")
    print(f"  huggingface-cli login")
    print(f"  huggingface-cli upload username/model-name ./{output}")


if __name__ == "__main__":
    main()
