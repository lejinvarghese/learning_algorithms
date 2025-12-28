"""
Evaluation utilities for Search Intention Network
"""

import torch
import click
from typing import List, Tuple


def score_candidates(
    model,
    prefix_text: str,
    candidate_texts: List[str],
    tokenizer,
    prefix_len: int = 20,
) -> List[Tuple[str, float]]:
    """Score candidate completions for a given prefix"""
    model.eval()
    with torch.no_grad():
        prefix_ids = tokenizer.encode(prefix_text, prefix_len).unsqueeze(0)

        # Move to model device
        device = next(model.parameters()).device
        prefix_ids = prefix_ids.to(device)

        # Score each candidate
        results = []
        for candidate_text in candidate_texts:
            candidate_ids = (
                tokenizer.encode(candidate_text, prefix_len).unsqueeze(0).to(device)
            )

            score = model(prefix_ids, candidate_ids)
            score_value = score.squeeze().item()
            results.append((candidate_text, score_value))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


def display_evaluation_results(
    prefix: str, scores: List[Tuple[str, float]], example_num: int = 1
):
    """Display evaluation results with colored bars"""
    click.secho(f"\n  Example {example_num}: '{prefix}'", fg="yellow")
    for j, (candidate, score) in enumerate(scores, 1):
        color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        click.secho(f"    {j}. [{bar}] {score:.4f} - {candidate}", fg=color)


def load_model_for_evaluation(checkpoint_path: str):
    """Load a trained model from checkpoint"""
    from train import SINLightningModule

    model = SINLightningModule.load_from_checkpoint(checkpoint_path)
    return model


def build_tokenizer(tokenizer_name: str = "google/byt5-small"):
    """Build ByT5 tokenizer"""
    from data import ByT5Tokenizer

    tokenizer = ByT5Tokenizer(model_name=tokenizer_name)
    return tokenizer


@click.command()
@click.option(
    "--checkpoint",
    default="./lightning_logs/sin_qac/version_5/checkpoints/final.ckpt",
    type=str,
    required=True,
    help="Path to model checkpoint",
)
@click.option(
    "--prefix", type=str, default="aline and pred", help="Prefix query to test"
)
@click.option(
    "--candidates",
    type=str,
    default="alien and predator,aliens,aliens & cowboys,x-men,aline",
    help="Comma-separated candidate completions",
)
@click.option(
    "--tokenizer_name",
    type=str,
    default="google/byt5-small",
    help="Tokenizer model name",
)
@click.option("--prefix_len", type=int, default=20, help="Prefix sequence length")
def main(checkpoint, prefix, candidates, tokenizer_name, prefix_len):
    """Evaluate trained model on prefix and candidate queries"""
    click.secho("\n🔍 Loading model and tokenizer...", fg="cyan", bold=True)

    # Parse comma-separated candidates
    candidates = [c.strip() for c in candidates.split(",")]

    try:
        model = load_model_for_evaluation(checkpoint)
        click.secho(f"✓ Model loaded from {checkpoint}", fg="green")
    except Exception as e:
        click.secho(f"❌ Failed to load checkpoint: {e}", fg="red")
        return

    try:
        tokenizer = build_tokenizer(tokenizer_name)
        click.secho(f"✓ Tokenizer loaded: {tokenizer.vocab_size} tokens", fg="green")
    except Exception as e:
        click.secho(f"❌ Failed to load tokenizer: {e}", fg="red")
        return

    # Score candidates
    click.secho(
        f"\n🎯 Scoring candidates for prefix: '{prefix}'", fg="bright_yellow", bold=True
    )
    click.secho("=" * 60, fg="bright_yellow")

    scores = score_candidates(
        model=model,
        prefix_text=prefix,
        candidate_texts=candidates,
        tokenizer=tokenizer,
        prefix_len=prefix_len,
    )

    # Display results
    click.secho("\n📊 Results (sorted by score):", fg="cyan", bold=True)
    for i, (candidate, score) in enumerate(scores, 1):
        bar_length = int(score * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        click.secho(f"\n  {i}. {candidate}", fg="white", bold=True)
        click.secho(f"     Score: {score:.4f}  {bar}", fg=color)

    click.secho("\n✅ Evaluation complete!\n", fg="bright_green", bold=True)


if __name__ == "__main__":
    main()
