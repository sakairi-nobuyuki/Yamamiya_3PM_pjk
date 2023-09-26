# coding: utf-8

<<<<<<< HEAD
import os

=======
>>>>>>> a16a2c25e0d9cf3b4d429dcef6baf6a64520a119
import typer

from ml_components.pipelines import TrainPipeline

app = typer.Typer()


<<<<<<< HEAD
@app.command("train")
def train(
    parapmeters: str = typer.Argument(
        ..., help="train parameters yaml path in cloud storage"
    )
):
    io_cofig = dict(
        endpoint_url=f"http://{os.environ['ENDPOINT_URL']}:9000",
        access_key=os.environ["ACCESS_KEY_ID"],
        secret_key=os.environ["SECRET_ACCESS_KEY"],
    )

    trainer = TrainPipeline(io_cofig, parapmeters)
    trainer.trainer.train()


@app.command("predict")
def predict_core(
    parameters_path: str = typer.Argument(
        ..., help="prediction parameters yaml path in local"
    )
):
    pass


def predict():
    pass
=======
@app.command()
def train():
    trainer = TrainPipeline()
    trainer.vgg_like.train()
>>>>>>> a16a2c25e0d9cf3b4d429dcef6baf6a64520a119


if __name__ == "__main__":
    app()
