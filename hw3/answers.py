r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 250
    hypers["seq_len"] = 128
    hypers["h_dim"] = 128
    hypers["n_layers"] = 3
    hypers["dropout"] = 0.2
    hypers["learn_rate"] = 0.01
    hypers["lr_sched_factor"] = 0.9
    hypers["lr_sched_patience"] = 1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I."
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences because training on the entire
Dataset create an exploding sum. when we Backpropagate through time
using a huge sequence we get a huge complicated computation graph.
this is why we split the corpus and using TBPTT (truncated back-propagation through time).

"""

part1_q2 = r"""
**Your answer:**

When generate the model we always! use the previous hidden state and output
as the next input to the model. but when training after every seq_len characters
we re-initialize the hidden-state.

"""

part1_q3 = r"""
**Your answer:**

We are not shuffling the order of the batches since the order
have a meaning in our data set.
The first sample in the first batch is the characters that come
right before the first sample in the second batch.
So in general the batches are continues and shuffling them will destroy
the real order of the text.

"""

part1_q4 = r"""
**Your answer:**

1. While sampling we might use lower temperature than 1 since
we would like for less uniform distribution because we want the model
to use what he learned in the training.
While training the model is still in training so using temperature 1.0
instead of a lower one will insure more uniform distribution and not just 
create the next char based on what he learned so far.

2. When the temperature is very high like 100. the sampling is 
a as uniform as it can gets. so we will get a random sequence of characters

3. when the temerature is very low such as 0.0001 we will get the same pattern
over and over again since we almost never allowing any characters other than the predicted one
to be printed.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=4, h_dim=125, z_dim=32, x_sigma2=0.01, learn_rate=0.0005, betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
1. When x_sigma2 is high (0.01) the model is not learning very well. 
The x_sigma2 is the variance of the encoder. 
When it's too high, there is a lot of noise and the latent space z is too big.
When x_sigma2 is lower, the generative images fits more to the data because
the latent space z is smaller. 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**
1. The purpose of the reconstruction loss is to make the prediction close to the original image.
The purpose of the KL-divergence loss is to make posterior distribution P(Z|x) 
close to the distribution of the data in the latent space P(Z).

2. the KL loss term minimize the weights such that the distribution of the latent space is 
close to N(0,I).

3. This effect is that sampling from the data will give as images 
from uniform distributed space.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**
We want to maximize the probability that the data that we receive can be generated from the model

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**
Because we want to make sure that the sigma is small, 
otherwise the sigma can be very large.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
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


part3_q1 = r"""
**Your answer:**
The training is built from two parts. 
When training the discriminator we want to sample in order to get a fake
picture for the classification, but we don't want to change the weights according to the 
gradient we get from the sampling, becouse in this stage of the learning 
the model learns to classify the images.
In the second part of training we want to improve the generator. In this case we want to keep
the gradient.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**
1. No. We also need to take into account the discriminator loss. 
If the discriminator is not accurate, the generator loss is useless, because it's
calculated with respect to the wrong labels.

2.It means that the model is still learning. The discriminator loss remains at a constant 
value, that means that the number of right guises of the model is constant (the accuracy of the model is nor dropping).
From the decrease in generator loss we get that more fake pictures is classified as a
real ones. 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**
The GAN generated images is not as good as the VAE generated images.
This is because the VAE generator learnes from real images, therefore it leanes fast. 
The GAN generator learns only from the discriminator labels, witch is much harder tast.  

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
