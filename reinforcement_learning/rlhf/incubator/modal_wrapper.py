import modal
import click

image = modal.Image.debian_slim(python_version="3.10").run_commands(
    "pip install click",
)
app = modal.App("example-get-started")


def func(x):
    click.secho("This code is running on a remote worker!", fg="magenta")
    return x**2


@app.function(image=image)
def funcx(x):
    return func(x)


@app.local_entrypoint()
def main():
    click.secho(f"the square is {funcx.local(42)}", fg="green")
    click.secho(f"the square is: {funcx.remote(42)}", fg="yellow")
