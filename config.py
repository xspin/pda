class Config:
    def __init__(self):
        self.image_size = (256, 256)
        self.batch_size = 16
        self.epochs = 3
        self.classes = 31
        self.base_model = 'vgg' # vgg or resnet

config = Config()