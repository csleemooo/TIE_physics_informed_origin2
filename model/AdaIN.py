from torch import nn

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
        self.mse_loss = nn.MSELoss()

    def calc_mean_std(self, feat, eps=1e-5):
        # ref: https://github.com/naoto0804/pytorch-AdaIN/blob/8af2958513117c83012ca9131da9333322934a3a/net.py#L95
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):

        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def calc_mean_std_channel(self, feat, eps=1e-5):
        # ref: https://github.com/naoto0804/pytorch-AdaIN/blob/8af2958513117c83012ca9131da9333322934a3a/net.py#L95
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C, H, W = size
        feat_var = feat.var(dim=1) + eps
        feat_std = feat_var.sqrt().view(N, 1, H, W)
        feat_mean = feat.mean(dim=1).view(N, 1, H, W)
        return feat_mean, feat_std

    def instance_normalization(self, feat):
        size = feat.size()
        m, s = self.calc_mean_std(feat)

        normalized_feat = (feat - m.expand(size)) / s.expand(size)
        return normalized_feat

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)