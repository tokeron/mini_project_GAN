import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    idx_to_char = {}
    sorted_text = sorted(text)
    char_to_idx = dict.fromkeys(sorted_text)
    for index, char in enumerate(char_to_idx):
        char_to_idx[char] = index
        idx_to_char[index] = char
    # for index, char in  char_to_idx:
    #     char :
    # char_to_idx = {}
    # idx_to_char = {}
    # for char in sorted_text:
    #     char_to_idx[char] = ord(char)
    #     idx_to_char[ord(char)] = char
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    text_clean = text
    for bad_char in chars_to_remove:
        text_clean = text_clean.replace(bad_char, "")
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    result = torch.zeros([len(text), len(char_to_idx)], dtype=torch.int8)
    result[range(len(text)), [char_to_idx[x] for x in text]] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    index = range(len(idx_to_char))
    index_vec = torch.CharTensor(index)
    result = ""
    string_indices = embedded_text @ index_vec
    for index in string_indices:
        result += idx_to_char[int(index)]
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    # char_to_concat = None
    # samples = None
    mod = len(text) % seq_len
    if mod > 0:
        char_to_concat = torch.tensor([char_to_idx[text[len(text)-mod]]], dtype=torch.int8)
        samples = torch.zeros([len(text[:len(text)-mod]), len(char_to_idx)], dtype=torch.int8)
        samples[range(len(text[:len(text) - mod])), [char_to_idx[x] for x in text[:len(text) - mod]]] = 1
    else:
        char_to_concat = torch.tensor([char_to_idx[text[len(text) - 1]]], dtype=torch.int8)
        samples = torch.zeros([len(text[:len(text) - 1]), len(char_to_idx)], dtype=torch.int8)
        samples[range(len(text[:len(text) - 1])), [char_to_idx[x] for x in text[:len(text) - 1]]] = 1
    index = range(len(char_to_idx))
    index_vec = torch.tensor(index, dtype=torch.int8)
    string_indices = samples @ index_vec
    string_indices = torch.cat((string_indices[1:len(string_indices)], char_to_concat))
    samples = samples.view([(len(text)-1)//seq_len, seq_len, len(char_to_idx)])
    labels = string_indices.view([(len(text)-1)//seq_len, seq_len])
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    scaled_y = y / temperature
    result = torch.softmax(scaled_y, dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    #  NOT GOOD YET!! @TODO

    # samples, labels = chars_to_labelled_samples(start_sequence, char_to_idx, start_sequence.len())
    # layer_prob, hidden_state = model.forward(samples)
    # layer_output = torch.multinomial(layer_prob, 1)
    # while layer_output[0].len() < n_chars:
    #     samples, labels = chars_to_labelled_samples(layer_output, char_to_idx, layer_output.len())
    #     additional_prob, next_hidden_state = model.forward(samples)
    #     additional_output = torch.multinomial(additional_prob, 1)
    #     layer_output += additional_output
    # return idx_to_char(layer_output)
    next_input = chars_to_onehot(start_sequence, char_to_idx)
    out, hidden_state = model.forward(next_input.unsqueeze(0).to(dtype=torch.float, device=device))
    #print("out shape generate_from_model: ", out.shape)
    #print("out from generate_from_model: ", out)
    prob = hot_softmax(out, dim=(len(out.shape)-1), temperature=T)
    # print("prob sum of label 1: ", prob.sum(dim=2))
    # print("prob shape: ", prob.shape)
    # print("prob: ", prob)
    #print("prob shape:",prob.shape)
    new_char = torch.multinomial(prob[-1][-1], 1)
    out_text += idx_to_char[new_char.item()]
    while len(out_text) < n_chars:
        next_input = chars_to_onehot(str(idx_to_char[new_char.item()]), char_to_idx)
        out, hidden_state = model.forward(next_input.unsqueeze(0).to(dtype=torch.float, device=device), hidden_state)
        prob = hot_softmax(out, dim=(len(out.shape)-1), temperature=T)
        new_char = torch.multinomial(prob[-1][-1], 1)
        out_text += idx_to_char[new_char.item()]
#  (B, S, I) where B is
#          the batch size, S is the length of each sequence and I is the
#          input dimension
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        idx = []
        dataset_len = len(self.dataset)
        number_of_batches = dataset_len // self.batch_size
        for counter in range(number_of_batches):
            idx = idx + list(range(counter, dataset_len - (dataset_len % self.batch_size), number_of_batches))
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        # self.layer_params.append(nn.Linear(in_dim, h_dim, bias=False))
        # self.add_module("input_lin", self.layer_params[-1])
        # for n in range(n_layers):
        #     lin = nn.Linear(h_dim, h_dim, bias=True)
        #     self.layer_params.append(lin)
        #     self.add_module("hid"+str(n+1), self.layer_params[-1])
        #     drop = nn.Dropout(dropout)
        #     self.layer_params.append(drop)
        #     self.add_module("drop"+str(n+1), self.layer_params[-1])
        # self.lin = nn.Linear(h_dim, out_dim, bias=True)
        # self.add_module("output_lin", self.lin)
        input_dim = self.in_dim
        self.is_drop = (dropout > 0)
        for n in range(n_layers):
            self.layer_params.append(nn.Linear(input_dim, h_dim, bias=False))
            self.add_module("xz" + str(n + 1), self.layer_params[-1])
            self.layer_params.append(nn.Linear(input_dim, h_dim, bias=False))
            self.add_module("xr" + str(n + 1), self.layer_params[-1])
            self.layer_params.append(nn.Linear(input_dim, h_dim, bias=False))
            self.add_module("xg" + str(n + 1), self.layer_params[-1])
            self.layer_params.append(nn.Linear(h_dim, h_dim, bias=True))
            self.add_module("hz" + str(n + 1), self.layer_params[-1])
            self.layer_params.append(nn.Linear(h_dim, h_dim, bias=True))
            self.add_module("hr" + str(n + 1), self.layer_params[-1])
            self.layer_params.append(nn.Linear(h_dim, h_dim, bias=True))
            self.add_module("hg" + str(n + 1), self.layer_params[-1])
            input_dim = h_dim
            if self.is_drop:
                self.layer_params.append(nn.Dropout(dropout))
                self.add_module("drop" + str(n + 1), self.layer_params[-1])
        self.layer_params.append(nn.Linear(h_dim, out_dim, bias=True))
        self.add_module("out", self.layer_params[-1])
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        output = []
        layer_input = layer_input.permute(1, 0, 2)
        for i in range(seq_len):
            t_input = layer_input[i]
            for j in range(self.n_layers):
                z = torch.sigmoid(self.__getattr__("xz"+str(j+1))(t_input) +
                                  self.__getattr__("hz"+str(j+1))(layer_states[j]))
                r = torch.sigmoid(self.__getattr__("xr"+str(j+1))(t_input) +
                                  self.__getattr__("hr"+str(j+1))(layer_states[j]))
                g = torch.tanh(self.__getattr__("xg"+str(j+1))(t_input) +
                               self.__getattr__("hg"+str(j+1))(r*layer_states[j]))
                layer_states[j] = z*layer_states[j] + (1-z)*g
                if self.is_drop:
                    t_input = self.__getattr__("drop"+str(j+1))(layer_states[j])
                else:
                    t_input = layer_states[j]
            output.append(self.out(layer_states[self.n_layers-1]))
        hidden_state = torch.stack(layer_states, dim=1)
        layer_output = torch.stack(output)
        layer_output = layer_output.permute(1, 0, 2)
        # ========================
        return layer_output, hidden_state
