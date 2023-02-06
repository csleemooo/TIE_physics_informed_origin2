from functions.Angular_Spectrum_Method import ASM
from torch import nn
import torch

class Holo_Generator(nn.Module):
    def __init__(self, args):
        super(Holo_Generator, self).__init__()
        self.wavelength = args.wavelength
        self.pixel_size = args.pixel_size
        self.distance_normalize = args.distance_max - args.distance_min
        self.distance_normalize_constant = args.distance_min / self.distance_normalize
        self.phase_normalize = args.phase_normalize


    def forward(self, amplitude, phase, d, return_field=False):

        d = ((d+self.distance_normalize_constant)*self.distance_normalize)

        phase = phase*self.phase_normalize
        O_low = amplitude*torch.exp(1j*phase)

        O_holo = ASM(O_low, self.wavelength, d, self.pixel_size, zero_padding=True)

        if return_field:
            return torch.abs(O_holo).float(), torch.angle(O_holo).float()
        else:
            return torch.pow(torch.abs(O_holo), 2).float()
