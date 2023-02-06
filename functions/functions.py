import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi, sqrt, floor
import torch.nn.functional as F
import torch
from skimage.restoration import unwrap_phase

def center_crop(H, size):
    batch, channel, Nh, Nw = H.size()

    return H[:, :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]

def random_crop(H, size):

    batch, channel, Nh, Nw = H.size()

    x_off = int(np.floor(np.random.rand() * (Nh-size)) + size/2)
    y_off = int(np.floor(np.random.rand() * (Nw-size)) + size/2)

    return H[:, :, (x_off - size//2) : (x_off+size//2), (y_off - size//2) : (y_off+size//2)]

def pos_crop(H, size, x_off, y_off):

    return H[:, :, (x_off - size//2) : (x_off+size//2), (y_off - size//2) : (y_off+size//2)]

def center_crop_numpy(H, size):
    Nh = H.shape[0]
    Nw = H.shape[1]

    return H[(Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]

def amp_pha_generate(real, imag):
    field = real + 1j*imag
    amplitude = np.abs(field)
    phase = np.angle(field)

    return amplitude, phase

def make_path(path):
    import os
    if not os.path.isdir(path):
        os.mkdir(path)


def save_fig(holo, fake_holo, real_amplitude, fake_amplitude, real_phase, fake_phase, real_distance, fake_distance, args, save_file):
    import matplotlib.pyplot as plt
    from math import pi
    fake_distance = fake_distance*args.distance_normalize
    fig2 = plt.figure(2, figsize=[12, 8])

    plt.subplot(2, 3, 1)
    plt.title('input holography')
    plt.imshow(holo, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.title('ground truth' + str(real_distance) + 'mm')
    plt.imshow(real_amplitude, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.title('output ' + str(np.round(fake_distance, 2)) + 'mm')
    plt.imshow(fake_amplitude, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.title('generated_holography')
    plt.imshow(fake_holo, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.title('ground truth phase')
    plt.imshow(real_phase, cmap='jet', vmax=pi, vmin=-pi)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.title('output phase')
    plt.imshow(fake_phase, cmap='jet', vmax=pi, vmin=-pi)
    plt.axis('off')
    plt.colorbar()

    fig2.savefig(save_file)
    plt.close(fig2)

def standardization(x):
    return (x-0.5)/0.5

def de_standardization(x):
    return (x+1)/2
def save_fig_distance(save_path, result_data, args):

    holo, fake_holo, real_amplitude, fake_amplitude, real_phase, fake_phase, real_distance, fake_distance= result_data
    real_phase -= np.mean(real_phase)
    fake_phase -= np.mean(fake_phase)
    fig2 = plt.figure(2, figsize=[12, 8])

    plt.subplot(2, 3, 1)
    plt.title('input holography')
    plt.imshow(holo, cmap='gray', vmax=0.5*args.intensity_rate, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.title('ground truth %dmm'%real_distance)
    plt.imshow(real_amplitude, cmap='gray', vmax=1*sqrt(args.intensity_rate), vmin=0)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.title('output ' + str(np.round(fake_distance, 2)) + 'mm')
    plt.imshow(fake_amplitude, cmap='gray', vmax=1*sqrt(args.intensity_rate), vmin=0)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.title('generated_holography')
    plt.imshow(fake_holo, cmap='gray', vmax=0.5*args.intensity_rate, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.title('ground truth phase')
    plt.imshow(real_phase, cmap='hot', vmax=2.5, vmin=-0.1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.title('output phase')
    plt.imshow(fake_phase, cmap='hot', vmax=2.5, vmin=-0.1)
    plt.axis('off')
    plt.colorbar()

    # fig_save_name = os.path.join(p, 'test' + str(b + 1) + '.png')
    fig2.savefig(save_path)
    plt.close(fig2)
def save_fig_pixel(save_path, result_data, args):

    holo, fake_holo, real_amplitude, fake_amplitude, real_phase, fake_phase, real_distance, fake_distance, real_pixel_scale, fake_pixel_scale = result_data

    fig2 = plt.figure(2, figsize=[12, 8])

    plt.subplot(2, 3, 1)
    plt.title('input holography %1.2f'%(real_pixel_scale))
    plt.imshow(holo, cmap='gray', vmax=0.5*args.intensity_rate, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.title('ground truth %dmm'%real_distance)
    plt.imshow(real_amplitude, cmap='gray', vmax=1*sqrt(args.intensity_rate), vmin=0)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.title('output ' + str(np.round(fake_distance, 2)) + 'mm')
    plt.imshow(fake_amplitude, cmap='gray', vmax=1*sqrt(args.intensity_rate), vmin=0)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.title('generated_holography %1.2f'%(fake_pixel_scale))
    plt.imshow(fake_holo, cmap='gray', vmax=0.5*args.intensity_rate, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.title('ground truth phase')
    plt.imshow(real_phase - np.mean(real_phase), cmap='hot', vmax=2.5, vmin=-0.1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.title('output phase')
    plt.imshow(fake_phase - np.mean(fake_phase), cmap='hot', vmax=2.5, vmin=-0.1)
    plt.axis('off')
    plt.colorbar()

    # fig_save_name = os.path.join(p, 'test' + str(b + 1) + '.png')
    fig2.savefig(save_path)
    plt.close(fig2)


def create_NA_circle(h,w,pixel_size, args):

    eff_pix = pixel_size/args.magnification  # scalar
    NA_radi = args.NA/args.wavelength
    max_fx = 1/(2*eff_pix)
    max_fy = 1/(2*eff_pix)

    x,y=torch.meshgrid(torch.linspace(start=-max_fx, end=max_fx, steps=h), torch.linspace(start=-max_fy, end=max_fy, steps=w), indexing='ij')
    C = torch.Tensor(((x**2 + y**2) <= NA_radi**2)*1.0).view(1,1,h,w)

    return C

import matplotlib.pyplot as plt
def Fourier_interpolation(amplitude, phase, scale_factor, N, args, crop_mode='random', x_off=0, y_off=0, crop_size=None):
    if amplitude.is_cuda:
        device='cuda'
    else:
        device='cpu'
    _,_, h, w = amplitude.shape
    pixel_scaled = args.pixel_size*scale_factor
    f_max_sclaed = 1/(pixel_scaled*2)
    f_max = 1/(args.pixel_size*2)
    f_range = np.linspace(start=-f_max, stop=f_max, num=h)
    f_scaled_N = np.sum(((f_range <= f_max_sclaed)*1) *((f_range>=-f_max_sclaed)*1))
    f_scaled_N = (f_scaled_N//2)*2
    # h_scaled, w_scaled = int(floor(h*scale_factor)//2*2), int(floor(w*scale_factor)//2*2)
    h_scaled, w_scaled = int(floor(h*scale_factor)//2*2), int(floor(w*scale_factor)//2*2)

    mean_amp = torch.mean(amplitude).item()

    # field = amplitude
    field = amplitude*torch.exp(1j*phase)
    img_fft = torch_fft(field)
    if args.pixel_size>pixel_scaled: # upsampling
        img_fft = F.pad(input=img_fft, value=0.0, mode='constant',
                        pad=((h_scaled-h)//2, (h_scaled-h)//2, (w_scaled-w)//2, (w_scaled-w)//2))

    elif args.pixel_size<pixel_scaled: # downsmapling
        img_fft = center_crop(img_fft, f_scaled_N)
    else:
        pass

    field = torch_ifft(img_fft)
    if crop_size:
        N=crop_size
    if crop_mode == 'random':
        field = random_crop(field, N)
    elif crop_mode == 'center':
        field = center_crop(field, N)
    else:
        field = pos_crop(field, N, x_off=x_off, y_off=y_off)

    NA = create_NA_circle(N, N, pixel_scaled, args)
    field=torch_ifft(torch_fft(field)*NA)

    amplitude = torch.abs(field)
    amplitude = amplitude*mean_amp/torch.mean(amplitude).item()
    phase = torch.angle(field)
    phase = phase-torch.mean(phase)

    return [amplitude, phase]


def torch_fft(H):

    H = torch.fft.fftshift(torch.fft.fft2(H), dim=(-2, -1))

    return H

def torch_ifft(H):

    H = torch.fft.ifft2(torch.fft.ifftshift(H, dim=(-2, -1)))

    return H
