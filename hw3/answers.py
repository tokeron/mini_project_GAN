# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams_vanilla():
    hypers = dict(
        batch_size=16,
        z_dim=32,
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0001,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.00001,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


def part3_gan_hyperparams_sngan():
    hypers = dict(
        batch_size=4,
        z_dim=64,
        data_label=1,
        label_noise=0.5,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.01    ,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.00001,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # ========================
    return hypers
# ==============

def part3_gan_hyperparams_wgan():
    hypers = dict(
        batch_size=16,
        z_dim=64,
        data_label=1,
        label_noise=0.5,
        discriminator_optimizer=dict(
            type="RMSprop",  # Any name in nn.optim like SGD, Adam
            lr=0.0001,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="RMSprop",  # Any name in nn.optim like SGD, Adam
            lr=0.00001,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # ========================
    return hypers
# ==============
