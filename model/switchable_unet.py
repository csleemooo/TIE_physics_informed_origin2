import torch
from torch import nn
from model.AdaIN import AdaIN

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

        return torch.sigmoid(out_d.view(-1, 1))

class code_generator_shared(nn.Module):
    def __init__(self, args):
        super(code_generator_shared, self).__init__()

        c1 = args.initial_channel
        c3 = args.initial_channel*8
        c5 = args.initial_channel*16
        self.code_G = nn.Sequential(nn.Linear(1, c1),
                                    nn.Linear(c1, c1),
                                    nn.Linear(c1, c3),
                                    nn.Linear(c3, c3),
                                    nn.Linear(c3, c5),
                                    nn.Linear(c5, c5))

    def forward(self, x):

        x = self.code_G(x)

        return x

class code_generator_layer(nn.Module):
    def __init__(self, args, out_channel):
        super(code_generator_layer, self).__init__()

        c5 = args.initial_channel*16

        self.fc_mean = nn.Linear(c5, out_channel)
        self.fc_var = nn.Linear(c5, out_channel)

        self.ReLU = nn.ReLU()

    def forward(self, input, shared_code):
        N, C, h, w = input.size()

        fc_mean = self.fc_mean(shared_code)
        fc_var = self.fc_var(shared_code)
        fc_var = self.ReLU(fc_var)

        fc_mean_np = fc_mean.view(N, C, 1, 1).expand(N, C, h, w)
        fc_var_np = fc_var.view(N, C, 1, 1).expand(N, C, h, w)

        return fc_mean_np, fc_var_np

class switchable_autoencoder(nn.Module):
    def __init__(self, args):
        super(switchable_autoencoder, self).__init__()

        self.encoder = switchable_encoder(args)
        self.decoder = switchable_decoder(args)
        self.distance_generator = Distance_Generator(args)

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, d_x, d_t, train=True):

        encoded = self.encoder(x)

        decoded_x = self.decoder(encoded, d_x)  # D(f, d)
        decoded_t = self.decoder(encoded, d_t)

        encoded_t = self.encoder(decoded_t)

        d_pred_x = self.distance_generator(x)
        d_pred_t = self.distance_generator(decoded_t)

        if train:
            loss_distance = self.mse_loss(d_pred_x, d_x) + self.mse_loss(d_pred_t, d_t)
            loss_identity = self.l1_loss(decoded_x, x)
            loss_content = self.mse_loss(encoded_t[-1], encoded[-1])

            return loss_identity, loss_content, loss_distance

        else:
            return decoded_x, decoded_t, d_pred_x, d_pred_t

class switchable_decoder(nn.Module):

    def __init__(self, args):
        super(switchable_decoder, self).__init__()

        c_list = [args.initial_channel * (2 ** i) for i in range(5)]

        self.l51 = one_conv_adain(args, c_list[4], c_list[4])
        self.l6 = up_conv(args, c_list[4], c_list[3])
        self.l7 = up_conv(args, c_list[3], c_list[2])
        self.l8 = up_conv(args, c_list[2], c_list[1])
        self.l9 = up_conv(args, c_list[1], c_list[0])
        self.conv_out = nn.Conv2d(c_list[0], 1, kernel_size=1, padding=0)

        self.shared_code = code_generator_shared(args)

    def forward(self, x, d):

        [x1, x2, x3, x4, x5] = x
        shared_code = self.shared_code(d)

        x = self.l51(x5, shared_code)
        x = self.l6(x, x4,shared_code)
        x = self.l7(x, x3, shared_code)
        x = self.l8(x, x2, shared_code)
        x = self.l9(x, x1, shared_code)

        x = self.conv_out(x)

        return x

class switchable_encoder(nn.Module):

    def __init__(self, args):
        super(switchable_encoder, self).__init__()

        c_list = [args.initial_channel*(2**i) for i in range(5)]
        self.l1 = down_conv(1, c_list[0])
        self.l2 = down_conv(c_list[0], c_list[1])
        self.l3 = down_conv(c_list[1], c_list[2])
        self.l4 = down_conv(c_list[2], c_list[3])
        self.l50 = one_conv(c_list[3], c_list[4])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.l1(x)
        x1_pool = self.pool(x1)
        x2 = self.l2(x1_pool)
        x2_pool = self.pool(x2)
        x3 = self.l3(x2_pool)
        x3_pool = self.pool(x3)
        x4 = self.l4(x3_pool)
        x4_pool = self.pool(x4)
        x5 = self.l50(x4_pool)

        return [x1, x2, x3, x4, x5]

class one_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(one_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class down_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()

        self.conv1 = one_conv(in_ch, out_ch)
        self.conv2 = one_conv(out_ch, out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class up_conv(nn.Module):
    def __init__(self, args, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch//2, kernel_size=2, stride=2)
        self.conv1 = one_conv(in_ch, out_ch)
        self.conv2 = one_conv_adain(args, out_ch, out_ch)

    def forward(self, x1, x2, shared_code_dec):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.conv2(x, shared_code_dec)
        return x

class one_conv_adain(nn.Module):
    def __init__(self, args, in_ch, out_ch):
        super(one_conv_adain, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.instanceNorm = nn.InstanceNorm2d(out_ch)
        self.adain = code_generator_layer(args, out_ch)
        self.Leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self,x_in, shared_code):
        x_in = self.conv(x_in)
        x_in = self.instanceNorm(x_in)

        N, C, h, w = x_in.size()
        mean_y, sigma_y = self.adain(x_in, shared_code)
        x_out = sigma_y * x_in + mean_y

        x_out = self.Leakyrelu(x_out)
        return x_out

def weights_initialize_normal(m):

    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:

        # apply a normal distribution to the weights and a bias=0
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.01)

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

        if self.lrelu:
            self.activation = nn.LeakyReLU(negative_slope=slope)
        else:
            self.activation = nn.ReLU()

    def forward(self, x):

        if self.use_norm:
            out = self.activation(self.Batch(self.Conv(x)))
        else:
            out = self.activation(self.Conv(x))

        return out