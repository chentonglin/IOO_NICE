import torch
import torch.nn as nn
import torch.nn.functional
from building_block import FC, EncoderConv, DecoderConv
from nice import *
import pdb
from torch.distributions.normal import Normal


def normalize_image(x):
    return x * 2 - 1


def restore_image(x):
    return (x + 1) * 0.5


def reparameterize_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    return mu + std * noise



class UpdaterBase(nn.Module):

    def __init__(self, channel_list, kernel_list, stride_list, hidden_list, num_planes_in, plane_height_in,
                 plane_width_in, num_features_out, state_size):
        super(UpdaterBase, self).__init__()
        self.net = EncoderConv(
            channel_list=channel_list,
            kernel_list=kernel_list,
            stride_list=stride_list,
            hidden_list=hidden_list,
            num_planes_in=num_planes_in,
            plane_height_in=plane_height_in,
            plane_width_in=plane_width_in,
            num_features_out=num_features_out,
            last_activation=True,
        )
        self.lstm = nn.LSTMCell(num_features_out, state_size)

    def forward(self, inputs, states=None):
        x = normalize_image(inputs)
        x = self.net(x)
        states = self.lstm(x, states)
        return states


class InitializerBack(UpdaterBase):

    def __init__(self, args):
        super(InitializerBack, self).__init__(
            channel_list=args.init_back_channel_list,
            kernel_list=args.init_back_kernel_list,
            stride_list=args.init_back_stride_list,
            hidden_list=args.init_back_hidden_list,
            num_planes_in=args.image_planes,
            plane_height_in=args.image_full_height,
            plane_width_in=args.image_full_width,
            num_features_out=args.init_back_size,
            state_size=args.image_planes*args.image_full_height*args.image_full_width,
        )


class InitializerFull(nn.Module):

    def __init__(self, args):
        super(InitializerFull, self).__init__()
        self.upd_main = UpdaterBase(
            channel_list=args.init_full_channel_list,
            kernel_list=args.init_full_kernel_list,
            stride_list=args.init_full_stride_list,
            hidden_list=args.init_full_main_hidden_list,
            num_planes_in=args.image_planes * 2 + 1,
            plane_height_in=args.image_full_height,
            plane_width_in=args.image_full_width,
            num_features_out=args.init_full_main_size,
            state_size=args.state_main_size,
        )
        self.net_full = FC(
            hidden_list=args.init_full_full_hidden_list,
            num_features_in=args.state_main_size,
            num_features_out=args.init_full_full_size,
            last_activation=True,
        )
        self.lstm_full = nn.LSTMCell(args.init_full_full_size, args.state_full_size)

    def forward(self, inputs, states_main):
        states_main = self.upd_main(inputs, states_main)
        x = self.net_full(states_main[0])
        states_full = self.lstm_full(x)
        return states_full, states_main


class InitializerCrop(nn.Module):

    def __init__(self, args):
        super(InitializerCrop, self).__init__()
        self.num_features_out = args.init_crop_size
        self.state_size1 = args.image_planes*args.image_crop_height*args.image_crop_width
        self.state_size2 = args.state_crop_size
        self.net = EncoderConv(
            channel_list=args.init_crop_channel_list,
            kernel_list=args.init_crop_kernel_list,
            stride_list=args.init_crop_stride_list,
            hidden_list=args.init_crop_hidden_list,
            num_planes_in=args.image_planes * 2 + 1,
            plane_height_in=args.image_crop_height,
            plane_width_in=args.image_crop_width,
            num_features_out=args.init_crop_size,
            last_activation=True,
        )
        self.lstm1 = nn.LSTMCell(self.num_features_out, self.state_size1)
        self.lstm2 = nn.LSTMCell(self.num_features_out, self.state_size2)

    def forward(self, inputs, states1=None, states2=None):
        x = normalize_image(inputs)
        x = self.net(x)
        states1 = self.lstm1(x, states1)
        states2 = self.lstm2(x, states2)
        return states1, states2
        


class UpdaterBack(UpdaterBase):

    def __init__(self, args):
        super(UpdaterBack, self).__init__(
            channel_list=args.upd_back_channel_list,
            kernel_list=args.upd_back_kernel_list,
            stride_list=args.upd_back_stride_list,
            hidden_list=args.upd_back_hidden_list,
            num_planes_in=args.image_planes * 3 + 1,
            plane_height_in=args.image_full_height,
            plane_width_in=args.image_full_width,
            num_features_out=args.upd_back_size,
            state_size=args.image_planes*args.image_full_height*args.image_full_width,
        )


class UpdaterFull(UpdaterBase):

    def __init__(self, args):
        super(UpdaterFull, self).__init__(
            channel_list=args.upd_full_channel_list,
            kernel_list=args.upd_full_kernel_list,
            stride_list=args.upd_full_stride_list,
            hidden_list=args.upd_full_hidden_list,
            num_planes_in=args.image_planes * 4 + 2,
            plane_height_in=args.image_full_height,
            plane_width_in=args.image_full_width,
            num_features_out=args.upd_full_size,
            state_size=args.state_full_size,
        )


class UpdaterCrop(nn.Module):

    def __init__(self, args):
        super(UpdaterCrop, self).__init__()
        self.num_features_out = args.init_crop_size
        self.state_size1 = args.image_planes*args.image_crop_height*args.image_crop_width
        self.state_size2 = args.state_crop_size
        self.net = EncoderConv(
            channel_list=args.upd_crop_channel_list,
            kernel_list=args.upd_crop_kernel_list,
            stride_list=args.upd_crop_stride_list,
            hidden_list=args.upd_crop_hidden_list,
            num_planes_in=args.image_planes * 4 + 2,
            plane_height_in=args.image_crop_height,
            plane_width_in=args.image_crop_width,
            num_features_out=args.upd_crop_size,
            last_activation=True,
        )

        self.lstm1 = nn.LSTMCell(self.num_features_out, self.state_size1)
        self.lstm2 = nn.LSTMCell(self.num_features_out, self.state_size2)

    def forward(self, inputs, states1, states2):
        x = normalize_image(inputs)
        x = self.net(x)
        states1 = self.lstm1(x, states1)
        states2 = self.lstm2(x, states2)
        return states1, states2

class EncDecBack(nn.Module):

    def __init__(self, args):
        super(EncDecBack, self).__init__()
        self.prior = torch.distributions.Normal( torch.tensor(0.), torch.tensor(1.))
        self.image_planes = args.image_planes
        self.image_full_height = args.image_full_height
        self.image_full_width = args.image_full_width
        self.flow = NICE(
            prior=self.prior, 
            coupling=2, 
            in_out_dim= self.image_planes*self.image_full_height*self.image_full_width,
             mid_dim=1000, 
             hidden=2, 
             mask_config=1,
            )  

    def encode(self, x):
        encoder_input = x.view(-1, self.image_planes*self.image_full_height*self.image_full_width)
        back_latent, _ = self.flow.f(encoder_input) 
        back_likelihood = self.flow(encoder_input)
        back_likelihood = back_likelihood.view(-1,1)
        return back_latent, back_likelihood

    def decode(self, back_latent):
        decoder_output = self.flow.g(back_latent)
        back = decoder_output.view(-1, self.image_planes, self.image_full_height, self.image_full_width) 
        return back
    #对背景进行编解码，输入输出维度为图片大小如（3,48,48），中间隐变量维度为(3*48*48）
    def forward(self, x):
        back_latent, back_likelihood = self.encode(x)
        back = self.decode(back_latent)
        result = {'back': back, 'back_likelihood': back_likelihood, 'back_latent': back_latent}
        return result


class EncDecFull(nn.Module):

    def __init__(self, args):
        super(EncDecFull, self).__init__()
        self.enc_pres = FC(
            hidden_list=args.enc_pres_hidden_list,
            num_features_in=args.state_full_size,
            num_features_out=3,
            last_activation=False,
        )
        self.enc_where = FC(
            hidden_list=args.enc_where_hidden_list,
            num_features_in=args.state_full_size,
            num_features_out=8,
            last_activation=False,
        )

    def encode(self, x):
        logits_tau1, logits_tau2, logits_zeta = self.enc_pres(x).chunk(3, dim=-1)
        tau1 = nn.functional.softplus(logits_tau1)
        tau2 = nn.functional.softplus(logits_tau2)
        zeta = torch.sigmoid(logits_zeta)
        where_mu, where_logvar = self.enc_where(x).chunk(2, dim=-1)
        return tau1, tau2, zeta, logits_zeta, where_mu, where_logvar

    @staticmethod
    def decode(where_mu, where_logvar):
        sample = reparameterize_normal(where_mu, where_logvar)
        scl = torch.sigmoid(sample[..., :2])
        trs = torch.tanh(sample[..., 2:])
        return scl, trs

    def forward(self, x):
        tau1, tau2, zeta, logits_zeta, where_mu, where_logvar = self.encode(x)
        scl, trs = self.decode(where_mu, where_logvar)
        result = {
            'scl': scl, 'trs': trs,
            'tau1': tau1, 'tau2': tau2, 'zeta': zeta, 'logits_zeta': logits_zeta,
            'where_mu': where_mu, 'where_logvar': where_logvar,
        }
        # print('scl:',scl.shape, 'trs:',trs.shape, 'taul:',tau1.shape, 'tau2:',tau2.shape, 'zeta:', zeta.shape, 'where_mu:', where_mu.shape)
        # pdb.set_trace()
        return result


class EncDecCrop(nn.Module):

    def __init__(self, args):
        super(EncDecCrop, self).__init__()
        self.prior = torch.distributions.Normal( torch.tensor(0.), torch.tensor(1.))
        self.image_planes = args.image_planes
        self.image_crop_height = args.image_crop_height
        self.image_crop_width = args.image_crop_width
        self.flow = NICE(
            prior=self.prior, 
            coupling=4, 
            in_out_dim= self.image_planes*self.image_crop_height*self.image_crop_width,
             mid_dim=1000, 
             hidden=5, 
             mask_config=1,
            )
        self.enc = FC(
            hidden_list=args.enc_what_hidden_list,
            num_features_in=args.state_crop_size,
            num_features_out=args.latent_what_size * 2,
            last_activation=False,
        )
        self.dec_shp = DecoderConv(
            channel_list_rev=args.dec_shp_channel_list_rev,
            kernel_list_rev=args.dec_shp_kernel_list_rev,
            stride_list_rev=args.dec_shp_stride_list_rev,
            hidden_list_rev=args.dec_shp_hidden_list_rev,
            num_features_in=args.latent_what_size,
            num_planes_out=1,
            plane_height_out=args.image_crop_height,
            plane_width_out=args.image_crop_width,
        )

        #x_apc的大小为（image_planes, image_crop_height, image_crop_width）
        #x_shp的大小为states_crop[0]
    def encode(self, x_apc, x_shp):
        encoder_input = x_apc.view(-1,self.image_planes*self.image_crop_height*self.image_crop_width)
        apc_latent, _ = self.flow.f(encoder_input)
        apc_likelihood = self.flow(encoder_input)
        apc_likelihood = apc_likelihood.view(-1,1)
        shp_mu, shp_logvar = self.enc(x_shp).chunk(2, dim=-1)
        return apc_latent, apc_likelihood, shp_mu, shp_logvar

    def decode(self, apc_latent, shp_mu, shp_logvar, grid_full):
        decoder_output = self.flow.g(apc_latent)
        apc_crop = decoder_output.view(-1, self.image_planes, self.image_crop_height, self.image_crop_width) #物体标准化的外观
        sample = reparameterize_normal(shp_mu, shp_logvar)
        logits_shp_crop = self.dec_shp(sample)
        shp_crop = torch.sigmoid(logits_shp_crop)  #物体标准化的形状
        apc = nn.functional.grid_sample(apc_crop, grid_full)   #物体的外观
        shp = nn.functional.grid_sample(shp_crop, grid_full)   #物体的外形状        
        return apc, shp, apc_crop, shp_crop

    #对背景进行编解码，输入输出维度为图片大小如（3,48,48），中间隐变量维度为(3*48*48）和
    def forward(self, x_apc, x_shp,  grid_full):
        apc_latent, apc_likelihood, shp_mu, shp_logvar = self.encode(x_apc, x_shp)
        apc, shp, apc_crop, shp_crop = self.decode(apc_latent, shp_mu, shp_logvar, grid_full)
        result = {
            'apc': apc, 'shp': shp, 'apc_crop': apc_crop, 'shp_crop': shp_crop,
            'shp_mu': shp_mu, 'shp_logvar': shp_logvar,'apc_latent': apc_latent, 'apc_likelihood':apc_likelihood,
        }
        return result
