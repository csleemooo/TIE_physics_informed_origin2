import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.use_norm = args.norm_use
        self.lrelu_use = args.lrelu_use
        self.batch_mode = args.batch_mode

        c1 = 16
        c2 = 64
        c3 = 256
        c4 = 64
        c5 = 16

        self.l10 = CBR(in_channel=1, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use)
        self.l11 = CBR(in_channel=c1, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use)

        self.l20 = CBR(in_channel=c1, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        self.l21 = CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)

        self.l30 = CBR(in_channel=c2, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        self.l31 = CBR(in_channel=c3, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)

        self.l40 = CBR(in_channel=c3, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        self.l41 = CBR(in_channel=c4, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)

        self.l50 = CBR(in_channel=c4, out_channel=c5, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)

        self.mpool0 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        l1 = self.l11(self.l10(x))
        l1_pool = self.mpool0(l1)

        l2 = self.l21(self.l20(l1_pool))
        l2_pool = self.mpool0(l2)

        l3 = self.l31(self.l30(l2_pool))
        l3_pool = self.mpool0(l3)

        l4 = self.l41(self.l40(l3_pool))
        l4_pool = self.mpool0(l4)

        l5_feat = self.l50(l4_pool)
        return l5_feat

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.use_norm = args.norm_use
        self.lrelu_use = args.lrelu_use
        self.batch_mode = args.batch_mode

        c1 = 16
        c2 = 32
        c3 = 64
        c4 = 128
        c5 = 256

        self.l51 = CBR(in_channel=c5, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        self.conv_T5 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l61 = CBR(in_channel=c4, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        self.l60 = CBR(in_channel=c3, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        # self.conv_T6 = nn.ConvTranspose2d(in_channels=c3, out_channels=c3, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.conv_T6 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l71 = CBR(in_channel=c3, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        self.l70 = CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        # self.conv_T7 = nn.ConvTranspose2d(in_channels=c2, out_channels=c2, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.conv_T7 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l81 = CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        self.l80 = CBR(in_channel=c2, out_channel=c1, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        # self.conv_T8 = nn.ConvTranspose2d(in_channels=c1, out_channels=c1, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.conv_T8 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l91 = CBR(in_channel=c1, out_channel=c1, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        self.l90 = CBR(in_channel=c1, out_channel=c1, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode)
        self.conv_out_holo = nn.Conv2d(in_channels=c1, out_channels=1, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        x = self.conv_T5(self.l51(x))
        x = self.conv_T6(self.l60(self.l61(x)))
        x = self.conv_T7(self.l70(self.l71(x)))
        x = self.conv_T8(self.l80(self.l81(x)))
        x = self.l90(self.l91(x))

        x = self.conv_out_holo(x)

        return x

class model_cvae(nn.Module):

    def __init__(self, args, input_channel=1, output_channel=2, z_dim=100):
        super(model_cvae, self).__init__()

        self.encoder_content = Encoder(args)
        self.encoder_distance = Encoder(args)
        self.decoder = Decoder(args)

        self.use_norm = args.norm_use
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.lrelu_use = args.lrelu_use
        self.batch_mode=args.batch_mode
        self.z_dim = z_dim

        c1 = args.initial_channel
        c2 = c1*2
        c3 = c2*2
        self.c3 = c3

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        if self.lrelu_use:
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        self.content_code_G = nn.Sequential(nn.Flatten(),
                                            nn.Linear(c3*16, c3*16),
                                            nn.BatchNorm1d(c3*16),
                                            self.activation,
                                            nn.Linear(c3*16, c3*16),
                                            nn.BatchNorm1d(c3*16),
                                            self.activation,
                                            nn.Linear(c3*16, z_dim))

        self.distance_code_G = nn.Sequential(nn.Flatten(),
                                            nn.Linear(c3 * 16, c3 * 16),
                                            nn.BatchNorm1d(c3 * 16),
                                            self.activation,
                                            nn.Linear(c3 * 16, c3 * 16),
                                            nn.BatchNorm1d(c3 * 16),
                                            self.activation,
                                            nn.Linear(c3 * 16, z_dim))

        self.latent = nn.Sequential(nn.Linear(z_dim*2, c3 * 16),
                                            nn.BatchNorm1d(c3 * 16),
                                            self.activation,
                                            nn.Linear(c3 * 16, c3 * 16),
                                            nn.BatchNorm1d(c3 * 16),
                                    self.activation)

        self.distance_cls = nn.Sequential(nn.Linear(z_dim, z_dim),
                                    nn.Linear(z_dim, z_dim),
                                          nn.Linear(z_dim, 1),
                                          nn.Sigmoid())

        self.A = torch.rand(z_dim, requires_grad=True).cuda()
        self.b = torch.rand(1, requires_grad=True).cuda()
        self.mse_loss = nn.MSELoss()

        self.apply(weights_initialize_normal)

    def forward(self, x, d=None, return_loss=False):

        content_feat = self.encoder_content(x)
        distance_feat = self.encoder_distance(x)

        content_code = self.content_code_G(content_feat)
        distance_code = self.distance_code_G(distance_feat)

        if d is not None:
            distance = d
        else:
            # distance = torch.sigmoid(torch.matmul(distance_code, self.A) + self.b)
            distance = self.distance_cls(distance_code)

        # distance_code_re = self.sigmoid_inverse(distance) - self.b
        # distance_code_re = torch.matmul(distance_code_re.view(-1, 1, 1), self.A.view(1, self.z_dim)).view(-1, self.z_dim)

        latent = torch.cat([content_code, distance_code], dim=1)
        b, _ = latent.shape
        latent = self.latent(latent).view(b, 16, 16, 16)

        x_re = self.decoder(latent)

        if return_loss:
            return x_re, distance #, self.mse_loss(distance_code, distance_code_re)
        else:
            return x_re, distance

    def sigmoid_inverse(self, x):
        return torch.log(x/((1-x)+1e-8))

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

