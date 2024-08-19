import wandb

class WandbLogger:

    def __init__(self, args) -> None:
        args.expname = str(args.index)
        self.run = wandb.init(project=args.project, config=args, name=args.expname)
        self.buffer = dict()

    def write(self, label: str, value) -> None:
        self.buffer[label] = value

    def write_img(self, label: str, img) -> None:
        img = wandb.Image(img, caption="mid")
        self.buffer[label] = img

    def write_imgs(self, label: str, imgs) -> None:
        wandbimgs = []
        for i, img in enumerate(imgs):
            wandbimgs.append(wandb.Image(img, caption=str(i)))
        self.buffer[label] = wandbimgs

    def update_buffer(self):
        self.run.log(self.buffer)
        self.buffer = dict()

    def write_checkpoint(self, path, expname):
        artifact = wandb.Artifact(name=f'checkpoint_{expname}', type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)
