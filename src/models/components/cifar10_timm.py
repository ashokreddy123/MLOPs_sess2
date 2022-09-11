from torch import nn
import timm

class cifar10_timm(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        img_size: int = 32,
    ):
        super().__init__()
        print("model_name",model_name)
        self.model = timm.create_model(model_name, img_size=img_size, pretrained='true')

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        #print("training is going on")
        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = cifar10_timm()
