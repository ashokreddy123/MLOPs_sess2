from torch import nn


class custom_model(nn.Module):
    def __init__(
        self,
        input_channesl: int = 3,
        hidden_layer1_channels: int = 64,
        hidden_layer2_channels: int = 128,
        hidden_layer3_channels: int = 256,
        hidden_layer4_channels: int = 256,
        flat1: int = 1024,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channesl, out_channels=hidden_layer1_channels,
			kernel_size=(3, 3)),
            nn.BatchNorm2d(hidden_layer1_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layer1_channels, out_channels=hidden_layer2_channels,
			kernel_size=(3, 3)),
            nn.BatchNorm2d(hidden_layer2_channels),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=hidden_layer2_channels, out_channels=hidden_layer3_channels,
			kernel_size=(3, 3)),
            nn.BatchNorm2d(hidden_layer3_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layer3_channels, out_channels=hidden_layer4_channels,
			kernel_size=(3, 3)),
            nn.BatchNorm2d(hidden_layer4_channels),
            nn.ReLU(),
            nn.Flatten(),

            nn.Linear(10*10*hidden_layer4_channels, flat1),
            nn.Linear(flat1, output_size),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        print("training is going on")
        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = custom_model()
