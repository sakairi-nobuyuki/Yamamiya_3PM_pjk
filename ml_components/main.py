# coding: utf-8

import typer

from ml_components.pipelines import TrainPipeline

app = typer.Typer()


@app.command()
def train():
    io_cofig = dict(endpoint_url="http://192.168.1.194:9000", access_key="sigma-chan", secret_key="sigma-chan-dayo")
    trainer = TrainPipeline(io_cofig)
    trainer.trainer.train()
    


if __name__ == "__main__":
    app()
