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
    args.save_name = 'switchable_unet'
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
    pretrain_distance_params = torch.load(os.path.join(args.model_root, args.project, 'tie_bead_training_pretrain_distance_generator', 'model.pth'))['model_state_dict']
    pretrain_distance_params = {i:j for i, j in pretrain_distance_params.items() if 'distance' in i}

    model = autoencoder(args).to(device)

    import copy
    for name, param in model.named_parameters():
        if name in pretrain_distance_params.keys():
            param.data = copy.copy(pretrain_distance_params[name])
            param.requires_grad = False

    model.distance_G.requires_grad_(False)

    propagator = Holo_Generator(args).to(device)

    op = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(op, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=args.epochs, eta_min=0)

    # target data
    loss_list = {'loss_sum_total': []}
    criterion = torch.nn.MSELoss().to(device)

    N_train = len(train_holo_loader)
    loss_sum_total, identity_loss_sum, distance_loss_sum = 0, 0, 0
    loss_sum_list=[]

    transform_simple = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomCrop(size=(args.crop_size, args.crop_size))])

    for epo in range(args.epochs):
        data_loader = iter(train_holo_loader)
        for batch in tqdm(range(N_train), desc='Iterations'):

            model.train()
            # model.distance_G.eval()
            # diff_intensity = torch.sqrt(next(data_loader))
            diff_intensity, d_true = next(data_loader)

            diff_intensity = diff_intensity.to(device).float()
            d_true = -args.distance_normalize_constant + d_true.view(-1, 1).to(device).float()/args.distance_normalize
            d_trans = torch.randint(low=0, high=17, size=[d_true.shape[0], 1]).to(device).float()/16

            loss_identity, loss_distance, out_holo_identity, out_holo_trans= model(diff_intensity, d_true, d_trans)

            op.zero_grad()

            loss_sum = args.w_identity*loss_identity + args.w_distance*loss_distance

            identity_loss_sum += args.w_identity*loss_identity.item()
            distance_loss_sum += args.w_distance*loss_distance.item()
            loss_sum_total += args.w_identity*loss_identity.item() + args.w_distance*loss_distance.item()

            loss_sum.backward()

            op.step()
            loss_sum_list.append(loss_sum.item())

            if (batch+1)%args.chk_iter == 0:
                print('[Epoch: %d] Total loss: %1.6f, Identity loss: %1.4f, Distance loss: %1.4f'
                      %(epo+1, loss_sum_total/args.chk_iter, identity_loss_sum/args.chk_iter, distance_loss_sum/args.chk_iter))
                loss_sum_total, identity_loss_sum, distance_loss_sum = 0, 0, 0

        else:
            loss_sum_total, identity_loss_sum, distance_loss_sum = 0, 0, 0
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
                    d_trans = torch.randint(low=0, high=17, size=[d_true.shape[0], 1]).to(device).float() / 16

                    diff_recon, d_pred, diff_recon_transform, _ = model(diff_intensity, d_true,
                                                                                            d_trans, train=False)


                    d_true = (d_true + args.distance_normalize_constant) * args.distance_normalize
                    d_pred = (d_pred + args.distance_normalize_constant) * args.distance_normalize
                    d_trans = (d_trans + args.distance_normalize_constant) * args.distance_normalize
                    diff_recon = diff_recon.cpu().detach().numpy()[0][0]
                    diff_intensity = diff_intensity.cpu().detach().numpy()[0][0]
                    diff_recon_transform = diff_recon_transform.cpu().detach().numpy()[0][0]

                    fig2 = plt.figure(2, figsize=[12, 8])
                    plt.subplot(1, 3, 1)
                    plt.title('True distance %1.6f' % (d_true.item()))
                    plt.imshow(diff_intensity, cmap='gray', vmax=1.0, vmin=0)
                    plt.axis('off')
                    plt.subplot(1, 3, 2)
                    plt.title('Pred distance %1.6f' % (d_pred.item()))
                    plt.imshow(diff_recon, cmap='gray', vmax=1.0, vmin=0)
                    plt.axis('off')
                    plt.subplot(1, 3, 3)
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