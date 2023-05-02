# coding: utf-8

import typer

from ml_components.pipelines import TrainPipeline

app = typer.Typer()


@app.command()
def train():
    trainer = TrainPipeline()
    trainer.vgg_like.train()


if __name__ == "__main__":
    app()
