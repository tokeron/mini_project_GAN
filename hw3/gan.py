import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from torch.nn.modules import conv
from torch.nn.modules.utils import _pair


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        nc = 3
        ndf = 64
        # modules.append(nn.Conv2d(in_channels=in_size[0], out_channels=64, kernel_size=(3, 3), ))
        modules.append(SNConv2d(nc, ndf, 3, 1, 1, bias=True)),
        modules.append(nn.LeakyReLU(0.1, inplace=True)),
        modules.append(SNConv2d(ndf, ndf, 4, 2, 1, bias=True)),
        modules.append(nn.LeakyReLU(0.1, inplace=True)),
        # state size. (ndf) x 1 x 32
        modules.append(SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True)),
        modules.append(nn.LeakyReLU(0.1, inplace=True)),
        modules.append(SNConv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=True)),
        # nn.BatchNorm2d(ndf * 2),
        modules.append(nn.LeakyReLU(0.1, inplace=True)),
        # state size. (ndf*2) x 16 x 16
        modules.append(SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True)),
        modules.append(nn.LeakyReLU(0.1, inplace=True)),
        modules.append(SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True)),
        modules.append(nn.LeakyReLU(0.1, inplace=True)),
        # state size. (ndf*8) x 4 x 4
        modules.append(SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True)),
        modules.append(nn.LeakyReLU(0.1, inplace=True)),
        modules.append(SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
        modules.append(nn.Sigmoid())

        # modules.append(nn.BatchNorm2d(num_features=64))
        # modules.append(nn.LeakyReLU())
        # modules.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=2, padding=2))
        # modules.append(nn.BatchNorm2d(num_features=128))
        # modules.append(nn.LeakyReLU())
        # modules.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3)))
        # modules.append(nn.BatchNorm2d(num_features=256))
        # modules.append(nn.LeakyReLU())
        # modules.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2))
        # modules.append(nn.BatchNorm2d(num_features=512))
        # modules.append(nn.LeakyReLU())
        # modules.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3)))
        # modules.append(nn.BatchNorm2d(num_features=1024))
        # modules.append(nn.LeakyReLU())
        # modules.append(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(12, 12)))
        self.cnn = nn.Sequential(*modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        # el = []
        # print(x.shape)
        # for i in range(x.shape[0]):
        #     el.append(self.cnn(x[0]))
        # y = torch.tensor(el)
        y = self.cnn(x)
        y = y.squeeze(-1)
        y = y.squeeze(-1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        self.featuremap_size = featuremap_size
        self.latent_lin = nn.Linear(in_features=z_dim, out_features=1024*featuremap_size*featuremap_size)

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []
        modules.append(nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=(9, 9)))
        # modules.append(nn.BatchNorm2d(num_features=512))
        # modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(3, 3)))
        modules.append(nn.BatchNorm2d(num_features=512))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=2))
        modules.append(nn.BatchNorm2d(num_features=256))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3)))
        modules.append(nn.BatchNorm2d(num_features=128))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, output_padding=1))
        modules.append(nn.BatchNorm2d(num_features=64))
        modules.append(nn.ReLU())
        modules.append(nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=(3, 3)))
        # modules.append(nn.BatchNorm2d(num_features=out_channels))
        modules.append(nn.Tanh())
        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        if with_grad:
            z = torch.normal(mean=0.0, std=1.0, size=(n, self.z_dim), device=device)
            samples = self.forward(z)
            return samples
        with torch.no_grad():
            z = torch.normal(mean=0.0, std=1.0, size=(n, self.z_dim), device=device)
            samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        batch_dim, z_dim = z.shape
        small_image = self.latent_lin(z)
        small_image = small_image.view(batch_dim, 1024, self.featuremap_size, self.featuremap_size)
        x = self.cnn(small_image)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = y_data.device
    N = y_data.shape[0]
    loss = torch.nn.BCEWithLogitsLoss()
    left = -label_noise/2
    right = -left
    data_labels = data_label + torch.FloatTensor(N,).uniform_(left, right).to(device)
    gen_labels = (1-data_label) + torch.FloatTensor(N,).uniform_(left, right).to(device)
    loss_data = loss(y_data, data_labels)
    loss_generated = loss(y_generated, gen_labels)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    N = y_generated.shape[0]
    device = y_generated.device
    loss_f = torch.nn.BCEWithLogitsLoss()
    data_labels = torch.tensor([data_label]*N, device=device, dtype=torch.float)
    loss = loss_f(y_generated, data_labels)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    batch_size = x_data.shape[0]


    gen_pic = gen_model.sample(batch_size, False)
    dsc_real_scores = dsc_model(x_data)
    dsc_fake_scores = dsc_model(gen_pic)
    dsc_real_scores = dsc_real_scores.squeeze(-1)
    dsc_fake_scores = dsc_fake_scores.squeeze(-1)
    dsc_loss = dsc_loss_fn(dsc_real_scores, dsc_fake_scores)
    dsc_optimizer.zero_grad()
    dsc_loss.backward()
    dsc_optimizer.step()

    dsc_real_scores = dsc_model(x_data)
    dsc_fake_scores = dsc_model(gen_pic)
    dsc_real_scores = dsc_real_scores.squeeze(-1)
    dsc_fake_scores = dsc_fake_scores.squeeze(-1)
    dsc_loss = dsc_loss_fn(dsc_real_scores, dsc_fake_scores)
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_pic = gen_model.sample(batch_size, True)
    gen_loss_scores = dsc_model(gen_pic)
    gen_loss_scores = gen_loss_scores.squeeze(-1)
    gen_loss = gen_loss_fn(gen_loss_scores)
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    gen_loss_scores = dsc_model(gen_pic)
    gen_loss_scores = gen_loss_scores.squeeze(-1)
    gen_loss = gen_loss_fn(gen_loss_scores)

    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if(len(dsc_losses) < 2 or dsc_losses[-1]+gen_losses[-1] < dsc_losses[-2]+gen_losses[-2]):
        saved = True
        saved_state = dict(
            dsc_loss=dsc_losses[-1],
            gen_loss=gen_losses[-1],
            model_state=gen_model.state_dict()
        )
        torch.save(saved_state, checkpoint_file)
    # ========================

    return saved


class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SNConv2d, self).__init__(
            in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            False, _pair(0), groups, bias, 'zeros')
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def Weights(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, u = max_singular_value(w_mat, self.u)
        self.u.copy_(u)
        return self.weight / sigma

    def forward(self, x):
        return F.conv2d(x, self.Weights, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


#define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)


def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u