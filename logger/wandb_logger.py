import wandb


class WandbLogger:

    def __init__(self, args) -> None:
        wandb.init(project="event-bad-nerf", config=args, name=args.expname)
        self.buffer = dict()

    def write(self, label: str, value) -> None:
        self.buffer[label] = value

    def write_img(self, label: str, img, caption=None) -> None:
        img = wandb.Image(img, caption=caption)
        self.buffer[label] = img

    def write_imgs(self, label: str, imgs, caption=None) -> None:
        wandbimgs = [wandb.Image(img, caption=caption) for img in imgs]
        self.buffer[label] = wandbimgs

    def update_buffer(self):
        wandb.log(self.buffer)

    def write_checkpoint(self, model):
        wandb.log_artifact(model)
