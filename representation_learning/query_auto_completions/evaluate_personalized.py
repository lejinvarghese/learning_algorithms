"""
Evaluation utilities for Personalized SIN
"""

import torch
import click
from typing import List, Tuple, Optional


def score_candidates_personalized(
    model,
    prefix_text: str,
    candidate_texts: List[str],
    tokenizer,
    prefix_len: int = 20,
    max_history_len: int = 10,
    history: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """Score candidate completions for a given prefix with history"""
    model.eval()
    history = history or []

    with torch.no_grad():
        device = next(model.parameters()).device
        prefix_ids = tokenizer.encode(prefix_text, prefix_len).unsqueeze(0).to(device)

        # Encode history
        history = history[-max_history_len:]
        history_ids_list = [tokenizer.encode(h, prefix_len) for h in history]
        num_hist = len(history_ids_list)
        while len(history_ids_list) < max_history_len:
            history_ids_list.append(torch.zeros(prefix_len, dtype=torch.long))
        history_ids = torch.stack(history_ids_list).unsqueeze(0).to(device)
        history_mask = torch.zeros(1, max_history_len, device=device)
        history_mask[0, :num_hist] = 1.0

        # Score each candidate
        results = []
        for candidate_text in candidate_texts:
            candidate_ids = (
                tokenizer.encode(candidate_text, prefix_len).unsqueeze(0).to(device)
            )
            score = model(prefix_ids, candidate_ids, history_ids, history_mask)
            score_value = score.squeeze().item()
            results.append((candidate_text, score_value))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


def display_evaluation_results(
    prefix: str,
    scores: List[Tuple[str, float]],
    example_num: int = 1,
    history: Optional[List[str]] = None,
):
    """Display evaluation results with colored bars"""
    click.secho(f"\n  Example {example_num}: '{prefix}'", fg="yellow")
    if history:
        click.secho(f"    History: {history}", fg="bright_black")
    for j, (candidate, score) in enumerate(scores, 1):
        color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        bar_len = int(score * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        click.secho(f"    {j}. [{bar}] {score:.4f} - {candidate}", fg=color)


def load_model_for_evaluation(checkpoint_path: str):
    """Load a trained personalized model from checkpoint"""
    from train_personalized import PersonalizedSINLightningModule

    model = PersonalizedSINLightningModule.load_from_checkpoint(checkpoint_path)
    return model


def build_tokenizer(tokenizer_name: str = "google/byt5-small"):
    """Build ByT5 tokenizer"""
    from data import ByT5Tokenizer

    return ByT5Tokenizer(model_name=tokenizer_name)


@click.command()
@click.option(
    "--checkpoint",
    default="./lightning_logs/sin_personalized/version_6/checkpoints/final.ckpt",
    type=str,
    required=True,
    help="Path to model checkpoint",
)
@click.option("--prefix", type=str, default="arma", help="Prefix query to test")
@click.option(
    "--candidates",
    type=str,
    default="armadillo,armageddon,armor",
    help="Comma-separated candidates",
)
@click.option("--history", type=str, default="", help="Comma-separated history queries")
@click.option(
    "--tokenizer_name",
    type=str,
    default="google/byt5-small",
    help="Tokenizer model name",
)
@click.option("--prefix_len", type=int, default=20, help="Prefix sequence length")
@click.option("--max_history_len", type=int, default=10, help="Max history length")
def main(
    checkpoint, prefix, candidates, history, tokenizer_name, prefix_len, max_history_len
):
    """Evaluate personalized model on prefix and candidate queries"""
    click.secho("\nLoading model and tokenizer...", fg="cyan", bold=True)

    candidates_list = [c.strip() for c in candidates.split(",")]
    history_list = [h.strip() for h in history.split(",") if h.strip()]

    try:
        model = load_model_for_evaluation(checkpoint)
        click.secho(f"Model loaded from {checkpoint}", fg="green")
    except Exception as e:
        click.secho(f"Failed to load checkpoint: {e}", fg="red")
        return

    try:
        tokenizer = build_tokenizer(tokenizer_name)
        click.secho(f"Tokenizer loaded: {tokenizer.vocab_size} tokens", fg="green")
    except Exception as e:
        click.secho(f"Failed to load tokenizer: {e}", fg="red")
        return

    # Score candidates
    click.secho(
        f"\nScoring candidates for prefix: '{prefix}'", fg="bright_yellow", bold=True
    )
    if history_list:
        click.secho(f"History: {history_list}", fg="cyan")
    click.secho("=" * 60, fg="bright_yellow")

    scores = score_candidates_personalized(
        model=model,
        prefix_text=prefix,
        candidate_texts=candidates_list,
        tokenizer=tokenizer,
        prefix_len=prefix_len,
        max_history_len=max_history_len,
        history=history_list,
    )

    # Display results
    click.secho("\nResults (sorted by score):", fg="cyan", bold=True)
    for i, (candidate, score) in enumerate(scores, 1):
        bar_length = int(score * 40)
        bar = "#" * bar_length + "." * (40 - bar_length)
        color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        click.secho(f"\n  {i}. {candidate}", fg="white", bold=True)
        click.secho(f"     Score: {score:.4f}  {bar}", fg=color)

    click.secho("\nEvaluation complete!\n", fg="bright_green", bold=True)


if __name__ == "__main__":
    main()
