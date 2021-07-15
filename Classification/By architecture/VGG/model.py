import torch
import torch.nn as nn

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, in_channels=3, height=224, width=224, num_classes=1000, architecture='VGG16'):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.num_classes = num_classes

        self.conv_layers = self.__init_conv_layers(architecture)
        self.flattening = nn.Flatten()
        self.fc_layers = self.__init_fc_layers(architecture)

    def __init_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        model_type = VGG_types[architecture]

        for x in model_type:
            if type(x) == int:
                out_channels = x
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ]
                )
                in_channels = x
            else:
                layers.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
        return nn.Sequential(*layers)

    def __init_fc_layers(self, architecture):
        model_type = VGG_types[architecture]
        pool_count = model_type.count('M')
        factor = 2 ** pool_count

        if (self.height % factor) + (self.width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )
        out_height = self.height // factor
        out_width = self.width // factor
        last_out_channels = next(
            x for x in model_type[::-1] if type(x) == int
        )
        return nn.Sequential(
            nn.Linear(last_out_channels * out_height * out_width, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flattening(x)
        x = self.fc_layers(x)
        return x
