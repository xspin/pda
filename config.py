class Config:
    def __init__(self):
        self.image_size = (256, 256)
        self.batch_size = 16
        self.epochs = 3
        self.classes = 31
        self.base_model = 'vgg' # vgg or resnet
        self.src_domain_name = 'amazon'
        self.tgt_domain_name = 'dslr'
        self.is_cuda = False

config = Config()


if __name__ == "__main__":
    print(config.__dict__)