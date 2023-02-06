import torch
from torch import nn
from model.AdaIN import AdaIN
class autoencoder(nn.Module):

    def __init__(self, args, input_channel=1, output_channel=2):
        super(autoencoder, self).__init__()

        self.use_norm = args.norm_use
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.lrelu_use = args.lrelu_use
        self.batch_mode=args.batch_mode

        c1 = args.initial_channel  # 32
        c2 = c1*2  # 64
        c3 = c2*2  # 128
        c4 = c3*2  # 256
        c5 = c4*2

        self.l10 = CBR(in_channel=self.input_channel, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use)
        self.l11 = CBR(in_channel=c1, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use)

        self.l20 = CBR(in_channel=c1, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l21 = CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.l30 = CBR(in_channel=c2, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l31 = CBR(in_channel=c3, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.l40 = CBR(in_channel=c3, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l41 = CBR(in_channel=c4, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.l50 = CBR(in_channel=c4, out_channel=c5, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.l5_distance = CBR(in_channel=c5, out_channel=1, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode, kernel=4, padding=1, stride=2)
        self.l5_content = CBR(in_channel=c5, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        # self.conv_T5 = nn.ConvTranspose2d(in_channels=c4, out_channels=c4, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.l51 = CBR(in_channel=c5, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T5 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l61 = CBR(in_channel=c5, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l60 = CBR(in_channel=c4, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        # self.conv_T6 = nn.ConvTranspose2d(in_channels=c3, out_channels=c3, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.conv_T6 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l71 = CBR(in_channel=c4, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l70 = CBR(in_channel=c3, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        # self.conv_T7 = nn.ConvTranspose2d(in_channels=c2, out_channels=c2, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.conv_T7 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l81 = CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l80 = CBR(in_channel=c2, out_channel=c1, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        # self.conv_T8 = nn.ConvTranspose2d(in_channels=c1, out_channels=c1, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.conv_T8 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l91 = CBR(in_channel=c1, out_channel=c1, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l90 = CBR(in_channel=c1, out_channel=c1, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.adain = AdaIN()
        self.distance_upsample = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                               CBR(in_channel=1, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode),
                                               CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode),
                                               nn.UpsamplingBilinear2d(scale_factor=2),
                                               CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode),
                                               CBR(in_channel=c2, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode),
                                               nn.UpsamplingBilinear2d(scale_factor=2),
                                               CBR(in_channel=c4, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode),
                                               CBR(in_channel=c4, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode),
                                               nn.UpsamplingBilinear2d(scale_factor=2))

        self.conv_out_holo = nn.Conv2d(in_channels=c1, out_channels=1, kernel_size=(1, 1), padding=0)

        self.apply(weights_initialize_normal)
        self.mpool0 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        if self.lrelu_use:
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

    def invertible_network_forward(self, x):

        return x

    def invertible_network_backward(self, x):
        return x

    def forward(self, x, d=None):

        l1 = self.l11(self.l10(x))
        l1_pool = self.mpool0(l1)

        l2 = self.l21(self.l20(l1_pool))
        l2_pool = self.mpool0(l2)

        l3 = self.l31(self.l30(l2_pool))
        l3_pool = self.mpool0(l3)

        l4 = self.l41(self.l40(l3_pool))
        l4_pool = self.mpool0(l4)

        l5_feat = self.l50(l4_pool)

        # generate distance
        l5_distance_feat = self.l5_distance(l5_feat)

        if d is not None:
            distance = d
        else:
            distance = self.sigmoid(self.avg_pool(l5_distance_feat))

        l5_content = self.l5_content(l5_feat)
        b, _,_,_ = l5_content.shape
        distance_expand = self.distance_upsample(distance)


        l5 = self.conv_T5(self.l51(torch.cat([distance_expand, l5_content], dim=1)))

        # l6 = l5
        l6 = torch.cat([l5, l4], dim=1)
        l6 = self.conv_T6(self.l60(self.l61(l6)))

        # l7 = l6
        l7 = torch.cat([l6, l3], dim=1)
        l7 = self.conv_T7(self.l70(self.l71(l7)))

        l8 = l7
        # l8 = torch.cat([l7, l2], dim=1)
        l8 = self.conv_T8(self.l80(self.l81(l8)))

        l9 = l8
        # l9 = torch.cat([l8, l1], dim=1)
        out = self.l90(self.l91(l9))

        out_holo = self.conv_out_holo(out)
        return out_holo, distance

class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1, use_norm=True, kernel=3, stride=1
                 , lrelu_use=False, slope=0.1, batch_mode='I', sampling='down', rate=1):
        super(CBR, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_norm = use_norm
        self.lrelu = lrelu_use

        if sampling == 'down':
            self.Conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(kernel, kernel), stride=(stride, stride),
                                  padding=padding, dilation=(rate, rate))
        else:
            self.Conv = nn.ConvTranspose2d(self.in_channel, self.out_channel, kernel_size=(2,2), stride=(2,2), padding=(0,0))

        if self.use_norm:
            if batch_mode == 'I':
                self.Batch = nn.InstanceNorm2d(self.out_channel)
            elif batch_mode == 'G':
                self.Batch = nn.GroupNorm(self.out_channel//16, self.out_channel)
            else:
                self.Batch = nn.BatchNorm2d(self.out_channel)

        self.lrelu = nn.LeakyReLU(negative_slope=slope)
        self.relu = nn.ReLU()

    def forward(self, x):

        if not self.lrelu:
            out = self.relu(self.Batch(self.Conv(x)))

        else:
            if self.use_norm:
                out = self.lrelu(self.Batch(self.Conv(x)))
            else:
                out = self.lrelu(self.Conv(x))

        return out

def weights_initialize_normal(m):

    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:

        # apply a normal distribution to the weights and a bias=0
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.01)

