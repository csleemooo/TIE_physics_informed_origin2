import os
import torch
import matplotlib
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from model.autoencoder import autoencoder
from model.Forward_model import Holo_Generator
from functions.data_loader import Holo_Recon_Dataloader
from functions.pretrain_argument import parse_args
from functions.functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# matplotlib.use('Agg')

if __name__ == '__main__':

    args = parse_args()
    args.distance_normalize = args.distance_max - args.distance_min
    args.distance_normalize_constant = args.distance_min / args.distance_normalize

    data_name_holo = "tie_bead_training_data"
    args.save_folder = data_name_holo + '_' + args.save_name

    args.project_path = os.path.join(args.model_root, args.project)
    make_path(args.project_path)
    args.saving_path = os.path.join(args.project_path, args.save_folder)
    make_path(args.saving_path)

    make_path(os.path.join(args.saving_path, 'generated'))

    args.data_path = os.path.join(args.data_root, data_name_holo)

    ## load test data
    transform_img = transforms.Compose([transforms.ToTensor()])
    train_holo_loader = Holo_Recon_Dataloader(root=args.data_path, data_type='holography',
                                              image_set='train', transform=transform_img, return_distance=True)
    train_holo_loader = DataLoader(train_holo_loader, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_holo_loader = Holo_Recon_Dataloader(root=args.data_path, data_type='holography',
                                                   image_set='test', transform=transform_img, return_distance=True)
    test_holo_loader=DataLoader(test_holo_loader, batch_size=1, shuffle=True, drop_last=True)

    N_test = test_holo_loader.__len__()

    # define model
    model = autoencoder(args).to(device)
    propagator = Holo_Generator(args).to(device)

    op = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(op, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=args.epochs, eta_min=0)

    # target data
    loss_list = {'loss_sum_total': []}
    criterion = torch.nn.MSELoss().to(device)

    N_train = len(train_holo_loader)
    loss_sum_total, ae_loss_sum, d_loss_sum = 0, 0, 0
    loss_sum_list=[]

    transform_simple = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomCrop(size=(args.crop_size, args.crop_size))])

    for epo in range(args.epochs):
        data_loader = iter(train_holo_loader)
        for batch in tqdm(range(N_train), desc='Iterations'):

            model.train()
            # diff_intensity = torch.sqrt(next(data_loader))
            diff_intensity, d_true = next(data_loader)

            diff_intensity = diff_intensity.to(device).float()
            d_true = -args.distance_normalize_constant + d_true.view(-1, 1, 1, 1).to(device).float()/args.distance_normalize

            diff_prop = propagator(torch.sqrt(diff_intensity), torch.zeros_like(diff_intensity).to(device).float(),
                                   -d_true-2*args.distance_normalize_constant)

            diff_recon, d_pred = model(diff_prop, d_true)

            ae_loss = criterion(diff_recon, diff_intensity)
            # d_loss = criterion(d_pred, d_true)

            op.zero_grad()

            loss_sum = ae_loss # + d_loss

            ae_loss_sum += ae_loss.item()
            #d_loss_sum += d_loss.item()
            loss_sum_total += ae_loss.item() #+ d_loss.item()

            loss_sum.backward()

            op.step()
            loss_sum_list.append(loss_sum.item())

            if (batch+1)%args.chk_iter == 0:
                print('[Epoch: %d] Total loss: %1.6f, AE loss: %1.4f'
                      %(epo+1, loss_sum_total/args.chk_iter, ae_loss_sum/args.chk_iter))
                loss_sum_total, ae_loss_sum, d_loss_sum = 0, 0, 0

        else:
            loss_sum_total, ae_loss_sum, d_loss_sum = 0, 0, 0
            lr_scheduler.step()

            if (epo+1)%args.model_save_iter==0:

                save_data = {'epoch': epo,
                             'model_state_dict': model.state_dict(),
                             'loss': loss_sum_list,
                             'args': args}

                torch.save(save_data, os.path.join(args.saving_path, "last_model_epo%d.pth"%(epo+1)))

            if (epo+1)%args.visualize_chk_iter==0:
                data_loader = iter(test_holo_loader)
                model.eval()

                # path for saving result
                p = os.path.join(args.saving_path, 'generated', 'epo%d' % (epo + 1))
                make_path(p)

                for b in range(N_test):
                    diff_intensity, d_true = next(data_loader)

                    diff_intensity = diff_intensity.to(device).float()
                    d_true = -args.distance_normalize_constant + d_true.view(-1, 1, 1, 1).to(
                        device).float() / args.distance_normalize

                    diff_prop = propagator(torch.sqrt(diff_intensity),
                                           torch.zeros_like(diff_intensity).to(device).float(),
                                           -d_true-2*args.distance_normalize_constant)

                    diff_recon, d_pred = model(diff_prop, d_true)

                    d_trans = torch.randperm(17)[0].view(1,1,1,1).to(device).float()/16
                    diff_recon_transform, _ = model(diff_prop, d_trans)


                    d_true = (d_true + args.distance_normalize_constant) * args.distance_normalize
                    d_pred = (d_pred + args.distance_normalize_constant) * args.distance_normalize
                    d_trans = (d_trans + args.distance_normalize_constant) * args.distance_normalize
                    diff_prop = diff_prop.cpu().detach().numpy()[0][0]
                    diff_recon = diff_recon.cpu().detach().numpy()[0][0]
                    diff_intensity = diff_intensity.cpu().detach().numpy()[0][0]
                    diff_recon_transform = diff_recon_transform.cpu().detach().numpy()[0][0]

                    fig2 = plt.figure(2, figsize=[12, 8])
                    plt.subplot(1, 4, 1)
                    plt.title('Network input')
                    plt.imshow(diff_prop, cmap='gray', vmax=1.0, vmin=0)
                    plt.axis('off')
                    plt.subplot(1, 4, 2)
                    plt.title('True distance %1.6f' % (d_true.item()))
                    plt.imshow(diff_intensity, cmap='gray', vmax=1.0, vmin=0)
                    plt.axis('off')
                    plt.subplot(1, 4, 3)
                    plt.title('Pred distance %1.6f' % (d_pred.item()))
                    plt.imshow(diff_recon, cmap='gray', vmax=1.0, vmin=0)
                    plt.axis('off')
                    plt.subplot(1, 4, 4)
                    plt.title('Trans distance %1.6f' % (d_trans.item()))
                    plt.imshow(diff_recon_transform, cmap='gray', vmax=1.0, vmin=0)
                    plt.axis('off')
                    plt.tight_layout()
                    fig_save_name = os.path.join(p, 'test_holo_' + str(b + 1) + '.png')
                    fig2.savefig(fig_save_name)
                    plt.close(fig2)


    else:
        save_data = {'epoch': args.epochs,
                     'model_state_dict': model.state_dict(),
                     'loss': loss_sum_list,
                     'args': args}

        torch.save(save_data, os.path.join(args.saving_path, "model.pth"))