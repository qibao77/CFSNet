import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from .modules.architecture import *
from .modules.loss import GANLoss, GradientPenaltyLoss

class CFSNetModel():
    def __init__(self, opt):
        # settings of train
        self.opt = opt
        train_opt = opt['train']
        self.input_alpha = opt['input_alpha']
        gpu_ids = opt['gpu_ids']
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        if self.is_train:
            self.use_gan = opt['use_gan']
            self.training_phase = train_opt['training_phase']
        self.schedulers = []
        self.optimizers = []

        # define network and load pretrained models
        opt_net_G = opt['network_G']
        self.netG = CFSNet(in_channel=opt_net_G['in_channel'], out_channel=opt_net_G['out_channel'],
                           num_channels=opt_net_G['num_channels'], num_main_blocks=opt_net_G['n_main_b'],
                           num_tuning_blocks=opt_net_G['n_tuning_b'], upscale=opt['scale'],
                           task_type=opt['task_type']).to(self.device)
        if gpu_ids:
            assert torch.cuda.is_available()
            self.netG = nn.DataParallel(self.netG)

        if self.is_train and self.use_gan:
            self.netD = define_D(opt).to(self.device) #Discriminator
            self.netD.train()

        self.load()
        self.save_dir = opt['path']['models']  # path to save the model

        if self.is_train:
            # define optimizer
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            if self.training_phase == 'main_branch':  # Step 1
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=train_opt['lr_G'], weight_decay=wd_G)
            else:  # Step 2
                if isinstance(self.netG, nn.DataParallel):
                    self.netG = self.netG.module
                for parm in self.netG.main.parameters():
                    parm.requires_grad = False
                self.optimizer_G = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.netG.tuning_blocks.parameters()), lr=train_opt['lr_G'],
                    weight_decay=wd_G)
            self.netG.train()
            self.optimizers.append(self.optimizer_G)

            # define loss
            # distortion loss
            if train_opt['loss_distortion_weight'] > 0:
                loss_dis_type = train_opt['loss_distortion_type']
                if loss_dis_type == 'l1':
                    self.loss_dis = nn.L1Loss().to(self.device)
                elif loss_dis_type == 'l2':
                    self.loss_dis = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss_dis_type))
                self.loss_dis_w = train_opt['loss_distortion_weight']
            else:
                print('Remove pixel loss.')
                self.loss_dis = None

            if self.use_gan:
                # perceptual loss
                if train_opt['loss_feature_weight'] > 0:
                    loss_fea_type = train_opt['loss_feature_type']
                    if loss_fea_type == 'l1':
                        self.loss_fea = nn.L1Loss().to(self.device)
                    elif loss_fea_type == 'l2':
                        self.loss_fea = nn.MSELoss().to(self.device)
                    else:
                        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss_fea_type))
                    self.loss_fea_w = train_opt['loss_feature_weight']
                else:
                    print('Remove feature loss.')
                    self.loss_fea = None
                if self.loss_fea:  # load VGG
                    self.netF = define_F(opt, use_bn=False).to(self.device)

                # gan loss
                self.loss_gan = GANLoss(train_opt['loss_gan_type'], 1.0, 0.0).to(self.device)
                self.loss_gan_w = train_opt['loss_gan_weight']
                # D_update_ratio and D_init_iters are for WGAN
                self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
                self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

                if train_opt['loss_gan_type'] == 'wgan-gp':
                    self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                    # gradient penalty loss
                    self.loss_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                    self.loss_gp_w = train_opt['gp_weigth']

                # D
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D']
                                                    , weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
                self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer,
                                                                    train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_ground_truth=True):
        if self.opt['task_type'] == 'denoise':
            self.gt_noise = torch.FloatTensor(data['GT'].size()).normal_(mean=0, std=self.opt['noise_level'] / 255.)
            self.Input = data['GT'] + self.gt_noise
            self.Input = self.Input.to(self.device)
            self.gt_noise = self.gt_noise.to(self.device)
        else:
            self.Input = data['LR'].to(self.device)  

        if need_ground_truth:
            self.ground_truth = data['GT'].to(self.device) 

    def optimize_parameters(self, step):
        # optimize G
        if self.training_phase == "tuning_branch":
            for parm in self.netG.main.parameters():
                parm.requires_grad = False
            for pa in self.netG.tuning_blocks.parameters():
                pa.requires_grad = True
        self.control_vector = torch.ones(self.Input.shape[0], 512) * self.input_alpha
        if self.opt['task_type'] == 'sr':
            self.control_vector = self.control_vector*0.5
        self.optimizer_G.zero_grad()
        self.Output = self.netG(x=self.Input, control_vector=self.control_vector)

        if self.use_gan:
            loss_total_g = 0
            if step % self.D_update_ratio == 0 and step >= self.D_init_iters:
                # Train G
                if self.loss_dis:  # distortion loss
                    loss_dis_g = self.loss_dis_w * self.loss_dis(self.Output, self.ground_truth)
                    loss_total_g = loss_dis_g
                if self.loss_fea:  # perceptual loss
                    real_fea = self.netF(self.ground_truth).detach()
                    fake_fea = self.netF(self.Output)
                    loss_fea_g = self.loss_fea_w * self.loss_fea(fake_fea, real_fea)
                    loss_total_g += loss_fea_g
                # gan loss
                pred_g_fake = self.netD(self.Output)
                loss_gan_g = -1*self.loss_gan_w * self.loss_gan(pred_g_fake)  # wgan:-E_g(D(x)), to min wasserstein distance:L=E_r(D(x))-E_g(D(x))
                loss_total_g += loss_gan_g

                loss_total_g.backward()
                self.optimizer_G.step()

            # Train D
            self.netD.zero_grad()
            loss_total_d = 0
            pred_d_real = self.netD(self.ground_truth)
            loss_gan_d_real = self.loss_gan(pred_d_real)  # wgan:-E_r(D(x))
            pred_d_fake = self.netD(self.Output.detach())
            loss_gan_d_fake = self.loss_gan(pred_d_fake)  # wgan:E_g(D(x))

            loss_total_d = loss_gan_d_fake-loss_gan_d_real

            if self.opt['train']['loss_gan_type'] == 'wgan-gp':
                alpha = torch.randn(self.Output.size(0), 1, 1, 1).to(self.device)
                interp = (alpha * self.ground_truth + ((1 - alpha) * self.Output.detach())).requires_grad_(True)
                interp_crit = self.netD(interp)

                loss_gp_d = self.loss_gp_w * self.loss_gp(interp, interp_crit)
                loss_total_d += loss_gp_d

            loss_total_d.backward(retain_graph=True)
            self.optimizer_D.step()

            # set log
            # D
            self.log_dict['loss_gan_d_real'] = loss_gan_d_real.item()
            self.log_dict['loss_gan_d_fake'] = loss_gan_d_fake.item()
            self.log_dict['loss_total_d'] = loss_total_d.item()

            if self.opt['train']['loss_gan_type'] == 'wgan-gp':
                self.log_dict['loss_gp_d'] = loss_gp_d.item()
            # D outputs
            self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
            self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())
            if step % self.D_update_ratio == 0 and step >= self.D_init_iters:
                # G
                if self.loss_dis:
                    self.log_dict['loss_dis_g'] = loss_dis_g.item()
                if self.loss_fea:
                    self.log_dict['loss_fea_g'] = loss_fea_g.item()
                self.log_dict['loss_gan_g'] = loss_gan_g.item()
        else:
            if self.opt['task_type'] == 'denoise':
                loss_dis = self.loss_dis_w * self.loss_dis(self.Output, self.gt_noise)
            else:
                loss_dis = self.loss_dis_w * self.loss_dis(self.Output, self.ground_truth)

            loss_dis.backward()
            self.optimizer_G.step()

            # set log
            self.log_dict['loss_dis'] = loss_dis.item()

    def test(self):
        self.netG.eval()
        self.control_vector = torch.ones(1, 512) * self.input_alpha
        with torch.no_grad():
            self.Output = self.netG(self.Input, control_vector=self.control_vector)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_ground_truth=True):
        out_dict = OrderedDict()
        out_dict['Input'] = self.Input.detach()[0].float().cpu()
        out_dict['Output'] = self.Output.detach()[0].float().cpu()
        if self.opt['task_type'] == 'denoise':
            out_dict['Output'] = out_dict['Input'] - out_dict['Output']
        if need_ground_truth:
            out_dict['ground_truth'] = self.ground_truth.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        network = self.netG
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))  # numle: get number of all elements
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

    def load(self):
        load_path_G = self.opt['path']['saved_model']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)
