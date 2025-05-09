from yaml import safe_load


class Config:
    def __init__(self, config_path: str) -> None:
        # global config
        # model config
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))
        self.pretrained_model_save = self.config["model"]["pretrained_model_save"]
        self.scheduler = self.config["model"]["scheduler"]

        # dataset config
        self.dataset_path = self.config["dataset"]["dataset_path"]

        # train config
        self.device = self.config["unet_train"]["device"]
        self.lr = float(self.config["unet_train"]["lr"])
        self.batch_size = int(self.config["unet_train"]["batch_size"])
        self.save_steps = int(self.config["unet_train"]["save_steps"])
        self.epochs = int(self.config["unet_train"]["epochs"])

        # backdoor config
        self.backdoor_mode = self.config["backdoor"]["mode"]
        self.num_badblocks = int(self.config["backdoor"]["num_badblocks"])

