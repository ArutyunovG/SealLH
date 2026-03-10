from torch import nn, Tensor

from typing import List

class ConvBnAct(nn.Module):

    def __init__(self, 
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 with_norm=True,
                 bias=True,
                 activation=nn.ReLU):

        super(ConvBnAct, self).__init__()

        stride = (stride, stride)
        dilation = (dilation, dilation)
        kernel_size = (kernel_size, kernel_size)
        conv_bias = bias and with_norm is False

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=conv_bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RepVggBlock(nn.Module):

    def __init__(self, ch_in, ch_out, act):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBnAct(ch_in, ch_out, 3, 1, padding=1, activation=nn.Identity)
        self.conv2 = ConvBnAct(ch_in, ch_out, 1, 1, padding=0, activation=nn.Identity)
        self.act = act()

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 num_blocks=3,
                 bias=None):
    
        super(UpBlock, self).__init__()
        self.conv1 = ConvBnAct(in_channels, hidden_channels, 1, 1, bias=bias, activation=nn.SiLU)
        self.conv2 = ConvBnAct(in_channels, hidden_channels, 1, 1, bias=bias, activation=nn.SiLU)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=nn.SiLU) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvBnAct(hidden_channels, out_channels, 1, 1, bias=bias, activation=nn.SiLU)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class RepFPN(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels
    ):
        super().__init__()

        n_blocks = 3
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_proj_p3 = ConvBnAct(in_channels[1], hidden_channels, 1, 1, activation=nn.SiLU)
        self.lateral_proj_p4 = ConvBnAct(in_channels[2], hidden_channels, 1, 1, activation=nn.SiLU)
        self.lateral_proj_p5 = ConvBnAct(in_channels[3], hidden_channels, 1, 1, activation=nn.SiLU)

        self.up_p3 = UpBlock(hidden_channels, out_channels, hidden_channels, num_blocks=n_blocks)
        self.up_p4 = UpBlock(hidden_channels, out_channels, hidden_channels, num_blocks=n_blocks)
        self.up_p5 = UpBlock(hidden_channels, out_channels, hidden_channels, num_blocks=n_blocks)

    def forward(self, features: List[Tensor]) -> List[Tensor]:

        [_, x2, x3, x4] = features

        l4 = self.lateral_proj_p5(x4)
        y4 = self.up_p5(l4)

        l3 = self.lateral_proj_p4(x3)
        l3 = l3 + self.upsample(y4)
        y3 = self.up_p4(l3)

        l2 = self.lateral_proj_p3(x2)
        l2 = l2 + self.upsample(y3)
        y2 = self.up_p4(l2)

        return [y2, y3, y4]
