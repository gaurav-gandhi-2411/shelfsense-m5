import typer

app = typer.Typer(
    name="shelfsense",
    help="ShelfSense M5 forecasting pipeline CLI.",
    no_args_is_help=True,
)

if __name__ == "__main__":
    app()
