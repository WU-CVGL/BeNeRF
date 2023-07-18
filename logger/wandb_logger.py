import wandb


class WandbLogger:

    def __init__(self, args) -> None:
        wandb.init(project=args.expname, config=args)
        self.buffer = dict()

    def write(self, label: str, value) -> None:
        self.buffer[label] = value

    def write_img(self, label: str, img, caption=None) -> None:
        img = wandb.Image(img, caption=caption)
        self.buffer[label] = img

    def update_buffer(self):
        wandb.log(self.buffer)
    def write_checkpoint(self, model):
        wandb.log_artifact(model)
