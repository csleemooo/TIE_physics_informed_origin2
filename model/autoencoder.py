import torch
from torch import nn
from model.AdaIN import AdaIN
class Discriminator(nn.Module):
    def __init__(self, args, input_channel=1):
        super(Discriminator, self).__init__()
        self.input_channel = input_channel
        self.output_channel = 1   # check ?!
        self.use_norm = True
        self.lrelu_use = args.lrelu_use
        self.batch_mode='B'

        # c1 = args.initial_channel
        c1 = args.initial_channel
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2

        self.l10 = CBR(in_channel=self.input_channel, out_channel=c1, use_norm=False, kernel=4, padding=0,
                       stride=2, lrelu_use=self.lrelu_use)

        self.l20 = CBR(in_channel=c1, out_channel=c2, use_norm=self.use_norm, kernel=4, padding=0, stride=2,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l30 = CBR(in_channel=c2, out_channel=c3, use_norm=self.use_norm, kernel=4, padding=0, stride=2,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l40 = CBR(in_channel=c3, out_channel=c4, use_norm=self.use_norm, kernel=4, padding=0, stride=1,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.conv_out = nn.Conv2d(in_channels=c4, out_channels=self.output_channel, kernel_size=(1, 1), stride=(1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.apply(weights_initialize_normal)

    def forward(self, x):

        x = self.l10(x)
        x = self.l20(x)
        x = self.l30(x)
        x = self.l40(x)

        out = self.conv_out(self.avg_pool(x))
        return out

class Distance_Generator(nn.Module):

    def __init__(self, args):
        super(Distance_Generator, self).__init__()
        self.input_channel = 1
        self.output_channel = 1
        self.lrelu_use = args.lrelu_use
        self.use_norm = True
        self.batch_mode = 'B'

        c1 = args.initial_channel
        c2 = c1*2
        c3 = c2*2

        # Stage 1
        self.l10 = CBR(in_channel=self.input_channel, out_channel=c1, use_norm=False, kernel=7, padding=3,
                       lrelu_use=self.lrelu_use)
        self.l11 = CBR(in_channel=c1, out_channel=c1, kernel=7, padding=3,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        # Stage 2
        self.l20 = CBR(in_channel=c1, out_channel=c2, kernel=5, padding=2,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l21 = CBR(in_channel=c2, out_channel=c2, kernel=5, padding=2,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.l30 = CBR(in_channel=c2, out_channel=c2, kernel=5, padding=2,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l31 = CBR(in_channel=c2, out_channel=c2, kernel=5, padding=2,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        # Stage 3
        self.l40 = CBR(in_channel=c2, out_channel=c3, kernel=3, padding=1,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l41 = CBR(in_channel=c3, out_channel=c3, kernel=3, padding=1,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.l50 = CBR(in_channel=c3, out_channel=c3, kernel=3, padding=1,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l51 = CBR(in_channel=c3, out_channel=c3, kernel=3, padding=1,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        # output
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_out_d = nn.Conv2d(in_channels=c3, out_channels=1, kernel_size=1)

        self.mpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.apply(weights_initialize_normal)

    def forward(self, x):

        l1 = self.mpool0(self.l11(self.l10(x)))
        l2 = self.mpool0(self.l21(self.l20(l1)))
        l3 = self.mpool0(self.l31(self.l30(l2)))
        l4 = self.mpool0(self.l41(self.l40(l3)))
        l5 = self.l51(self.l50(l4))

        out_d = self.global_avg_pool(l5)
        out_d = self.conv_out_d(out_d)

        return out_d.view(-1, 1)

class autoencoder(nn.Module):

    def __init__(self, args, input_channel=1, output_channel=2):
        super(autoencoder, self).__init__()

        self.use_norm = args.enc_norm_use
        self.dec_use_norm = args.dec_norm_use
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.lrelu_use = args.lrelu_use
        self.batch_mode=args.batch_mode
        self.z_dim = 100

        self.distance_G = Distance_Generator(args)

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
        self.l51 = CBR(in_channel=c5, out_channel=c4, use_norm=self.dec_use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T5 = nn.ConvTranspose2d(in_channels=c4, out_channels=c4, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        # self.conv_T5 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l61 = CBR(in_channel=c5, out_channel=c4, use_norm=self.dec_use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l60 = CBR(in_channel=c4, out_channel=c3, use_norm=self.dec_use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T6 = nn.ConvTranspose2d(in_channels=c3, out_channels=c3, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        # self.conv_T6 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l71 = CBR(in_channel=c4, out_channel=c3, use_norm=self.dec_use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l70 = CBR(in_channel=c3, out_channel=c2, use_norm=self.dec_use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T7 = nn.ConvTranspose2d(in_channels=c2, out_channels=c2, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        # self.conv_T7 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l81 = CBR(in_channel=c3, out_channel=c2, use_norm=self.dec_use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l80 = CBR(in_channel=c2, out_channel=c1, use_norm=self.dec_use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T8 = nn.ConvTranspose2d(in_channels=c1, out_channels=c1, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        # self.conv_T8 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.l91 = CBR(in_channel=c2, out_channel=c1, use_norm=self.dec_use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l90 = CBR(in_channel=c1, out_channel=c1, use_norm=self.dec_use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.adain = AdaIN()

        self.conv_out_holo = nn.Conv2d(in_channels=c1, out_channels=1, kernel_size=(1, 1), padding=0)

        self.mpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        if self.lrelu_use:
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        self.relu = nn.ReLU()
        self.shared_code_generator = nn.Sequential(nn.Linear(100, 128),
                                                   nn.Linear(128, 128),
                                                   nn.Linear(128, 128),
                                                   nn.Linear(128, 128),
                                                   nn.Linear(128, 128),
                                                   nn.Linear(128, 128),
                                                   nn.Linear(128, 128),
                                                   nn.Linear(128, 128))

        [self.l5_code_gm, self.l5_code_gs] = self.build_code_g(c5)
        [self.l6_code_gm, self.l6_code_gs] = self.build_code_g(c4)
        [self.l7_code_gm, self.l7_code_gs] = self.build_code_g(c3)
        [self.l8_code_gm, self.l8_code_gs] = self.build_code_g(c2)
        [self.l9_code_gm, self.l9_code_gs] = self.build_code_g(c1)

        self.mse_loss = nn.MSELoss()
        self.apply(weights_initialize_normal)

    def build_code_g(self, out_ch):
        m_g = nn.Linear(128, out_ch)
        s_g = nn.Linear(128, out_ch)

        return [m_g, s_g]

    def forward_code_g(self, x, code, g0, g1):
        N,C, h, w = x.shape

        fc_mean = g0(code)
        fc_var = self.relu(g1(code))

        fc_mean_np = fc_mean.view(N, C, 1, 1).expand(N, C, h, w)
        fc_var_np = fc_var.view(N, C, 1, 1).expand(N, C, h, w)

        x_mean, x_var = self.adain.calc_mean_std(x)

        x = (x - x_mean.expand(N, C, h, w)) / x_var.expand(N, C, h, w)
        x = x*fc_var_np + fc_mean_np

        return x

    def forward_IN(self, x):
        N,C,h,w = x.shape
        x_mean, x_var = self.adain.calc_mean_std(x)

        x = (x - x_mean.expand(N, C, h, w)) / x_var.expand(N, C, h, w)
        return x


    def forward_decoder(self, x, skip, distance_code):

        l5 = self.conv_T5(self.l51(self.forward_code_g(x, distance_code, self.l5_code_gm, self.l5_code_gs)))

        # l6 = l5
        l6 = self.l61(torch.cat([l5, skip[0]], dim=1))
        l6 = self.conv_T6(self.l60(self.forward_code_g(l6, distance_code, self.l6_code_gm, self.l6_code_gs)))
        # l6 = self.l60(self.forward_code_g(l6, distance_code, self.l6_code_gm, self.l6_code_gs))

        # l7 = l6
        l7 = self.l71(torch.cat([l6, skip[1]], dim=1))
        l7 = self.conv_T7(self.l70(self.forward_code_g(l7, distance_code, self.l7_code_gm, self.l7_code_gs)))

        # l8 = l7
        l8 = self.l81(torch.cat([l7, skip[2]], dim=1))
        l8 = self.conv_T8(self.l80(self.forward_code_g(l8, distance_code, self.l8_code_gm, self.l8_code_gs)))
        # l8 =self.l80(self.forward_code_g(l8, distance_code, self.l8_code_gm, self.l8_code_gs))

        # l9 = l8
        l9 = self.l91(torch.cat([l8, skip[3]], dim=1))
        out = self.l90(self.forward_code_g(l9, distance_code, self.l9_code_gm, self.l9_code_gs))

        out_holo = self.conv_out_holo(out)

        return out_holo

    def forward_encoder(self, x):
        # encoder part
        l1 = self.l11(self.l10(x))
        l1_pool = self.mpool0(l1)
        # l1_pool = l1

        l2 = self.l21(self.l20(l1_pool))
        l2_pool = self.mpool0(l2)

        l3 = self.l31(self.l30(l2_pool))
        l3_pool = self.mpool0(l3)
        # l3_pool = l3

        l4 = self.l41(self.l40(l3_pool))
        l4_pool = self.mpool0(l4)

        encoded = self.l50(l4_pool)

        return [l4, l3, l2, l1], encoded

    def forward(self, x, d_true, d_trans, train=True):

        assert d_true.shape == d_trans.shape
        b, _ = d_true.shape

        # code generator
        # d_true_vec = d_true
        d_true_vec = torch.rand(size=[b, self.z_dim]).to(d_true.device)*0.1 + d_true.expand(b, self.z_dim)
        d_true_code = self.shared_code_generator(d_true_vec)

        # d_trans_vec = d_trans
        d_trans_vec = torch.rand(size=[b, self.z_dim]).to(d_trans.device)*0.1 + d_trans.expand(b, self.z_dim)
        d_trans_code = self.shared_code_generator(d_trans_vec)

        # encoder
        skip, encoded = self.forward_encoder(x)

        # identity reconstruction
        out_holo_identity = self.forward_decoder(encoded, skip, d_true_code)

        # transformation
        out_holo_trans = self.forward_decoder(encoded, skip, d_trans_code)
        _, trans_encoded = self.forward_encoder(out_holo_trans)

        distance = self.distance_G(out_holo_identity)
        distance_trans = self.distance_G(out_holo_trans)

        if train:
            loss_identity = self.mse_loss(out_holo_identity, x)
            # loss_identity = self.mse_loss(self.forward_IN(trans_encoded), self.forward_IN(encoded))
            loss_distance = self.mse_loss(distance, d_true) + self.mse_loss(distance_trans, d_trans)

            return loss_identity, loss_distance, out_holo_identity, out_holo_trans

        else:
            return out_holo_identity, distance, out_holo_trans, distance_trans

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
                self.Batch = nn.GroupNorm(self.out_channel//16, self.ou555t_channel)
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

