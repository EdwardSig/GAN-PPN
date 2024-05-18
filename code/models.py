import sys

import numpy as np
import torch
from torch import nn

class VGANNonPop():
    def __init__(self, nb_user, nb_item, neg_dis_size, pos_dis_size, discriminator_vector_dim, generator_vector_dim, device):
        self.D = Discriminator(nb_item, neg_dis_size, discriminator_vector_dim)
        self.G = Generator(nb_item, pos_dis_size, generator_vector_dim)

        self.neg_distribution_vector = []       # 针对不同用户的负样本兴趣分布的向量
        self.pos_distribution_vector = []       # 针对不同用户的正样本兴趣分布的向量

        for u in range(0, nb_user):
            self.neg_distribution_vector.append(np.random.normal(loc=0, scale=1, size=discriminator_vector_dim).tolist())
            self.pos_distribution_vector.append(np.random.normal(loc=0, scale=1, size=generator_vector_dim).tolist())
        self.neg_distribution_vector = torch.FloatTensor(self.neg_distribution_vector).to(device)
        self.pos_distribution_vector = torch.FloatTensor(self.pos_distribution_vector).to(device)


class Discriminator(nn.Module):
    def __init__(self, nb_item, neg_dis_size, vector_dim):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(nb_item + neg_dis_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.mean_vector = nn.Linear(neg_dis_size, vector_dim)
        self.std_vector = nn.Linear(neg_dis_size, vector_dim)
        self.decoder = nn.Linear(vector_dim, nb_item)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.mean_vector.weight, std=0.01)
        nn.init.normal_(self.std_vector.weight, std=0.01)
        nn.init.normal_(self.decoder.weight, std=0.01)

    def forward(self, data, neg_distribution_data, neg_distribution_vector):
        fix_data = data + neg_distribution_data
        data = torch.cat([data, neg_distribution_data], dim=-1)
        out = self.dis(data)
        z = (torch.exp(self.std_vector(fix_data)) * neg_distribution_vector) + self.mean_vector(fix_data)
        neg_distribution_pred = self.decoder(z)

        return out, neg_distribution_pred

    def get_z_distribution(self, neg_distribution_data, neg_distribution_vector):
        z = (torch.exp(self.std_vector(neg_distribution_data)) * neg_distribution_vector) + self.mean_vector(neg_distribution_data)

        return z


class Generator(nn.Module):
    def __init__(self, nb_item, pos_dis_size, vector_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(nb_item + pos_dis_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, nb_item),
            nn.ReLU()
        )
        self.mean_vector = nn.Linear(pos_dis_size, vector_dim)
        self.std_vector = nn.Linear(pos_dis_size, vector_dim)
        self.decoder = nn.Linear(vector_dim, nb_item)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.mean_vector.weight, std=0.01)
        nn.init.normal_(self.std_vector.weight, std=0.01)
        nn.init.normal_(self.decoder.weight, std=0.01)

    def forward(self, data, pos_distribution_data, pos_distribution_vector):
        fix_data = data + pos_distribution_data
        data = torch.cat([data, pos_distribution_data], dim=-1)
        out = self.gen(data)
        z = (torch.exp(self.std_vector(fix_data)) * pos_distribution_vector) + self.mean_vector(fix_data)
        pos_distribution_pred = self.decoder(z)
        return out, pos_distribution_pred

    def get_z_distribution(self, pos_distribution_data, pos_distribution_vector):
        z = (torch.exp(self.std_vector(pos_distribution_data)) * pos_distribution_vector) + self.mean_vector(pos_distribution_data)

        return z


class DiscriminatorNonNegativePref(nn.Module):
    def __init__(self, nb_item, neg_dis_size, vector_dim):
        super(DiscriminatorNonNegativePref, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(nb_item, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.mean_vector = nn.Linear(neg_dis_size, vector_dim)
        self.std_vector = nn.Linear(neg_dis_size, vector_dim)
        self.decoder = nn.Linear(vector_dim, nb_item)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.mean_vector.weight, std=0.01)
        nn.init.normal_(self.std_vector.weight, std=0.01)
        nn.init.normal_(self.decoder.weight, std=0.01)

    def forward(self, data, neg_distribution_vector):
        fix_data = data
        out = self.dis(data)
        z = (torch.exp(self.std_vector(fix_data)) * neg_distribution_vector) + self.mean_vector(fix_data)
        neg_distribution_pred = self.decoder(z)

        return out, neg_distribution_pred

    def get_z_distribution(self, neg_distribution_data, neg_distribution_vector):
        z = (torch.exp(self.std_vector(neg_distribution_data)) * neg_distribution_vector) + self.mean_vector(neg_distribution_data)

        return z


class GeneratorNonPositivePref(nn.Module):

    def __init__(self, nb_item, pos_dis_size, vector_dim):

        super(GeneratorNonPositivePref, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(nb_item, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, nb_item),
            nn.ReLU()
        )
        self.mean_vector = nn.Linear(pos_dis_size, vector_dim)
        self.std_vector = nn.Linear(pos_dis_size, vector_dim)
        self.decoder = nn.Linear(vector_dim, nb_item)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.mean_vector.weight, std=0.01)
        nn.init.normal_(self.std_vector.weight, std=0.01)
        nn.init.normal_(self.decoder.weight, std=0.01)

    def forward(self, data, pos_distribution_vector):

        fix_data = data
        out = self.gen(data)
        z = (torch.exp(self.std_vector(fix_data)) * pos_distribution_vector) + self.mean_vector(fix_data)
        pos_distribution_pred = self.decoder(z)
        return out, pos_distribution_pred

    def get_z_distribution(self, pos_distribution_data, pos_distribution_vector):

        z = (torch.exp(self.std_vector(pos_distribution_data)) * pos_distribution_vector) + self.mean_vector(pos_distribution_data)

        return z


class VGANNonNegativePref():
    def __init__(self, nb_user, nb_item, neg_dis_size, pos_dis_size, discriminator_vector_dim, generator_vector_dim, device):

        self.D = DiscriminatorNonNegativePref(nb_item, neg_dis_size, discriminator_vector_dim)
        self.G = Generator(nb_item, pos_dis_size, generator_vector_dim)

        self.neg_distribution_vector = []
        self.pos_distribution_vector = []

        for u in range(0, nb_user):
            self.neg_distribution_vector.append(np.random.normal(loc=0, scale=1, size=discriminator_vector_dim).tolist())
            self.pos_distribution_vector.append(np.random.normal(loc=0, scale=1, size=generator_vector_dim).tolist())
        self.neg_distribution_vector = torch.FloatTensor(self.neg_distribution_vector).to(device)
        self.pos_distribution_vector = torch.FloatTensor(self.pos_distribution_vector).to(device)


class VGANNonPositivePref():
    def __init__(self, nb_user, nb_item, neg_dis_size, pos_dis_size, discriminator_vector_dim, generator_vector_dim, device):
        self.D = Discriminator(nb_item, neg_dis_size, discriminator_vector_dim)
        self.G = GeneratorNonPositivePref(nb_item, pos_dis_size, generator_vector_dim)

        self.neg_distribution_vector = []
        self.pos_distribution_vector = []

        for u in range(0, nb_user):
            self.neg_distribution_vector.append(np.random.normal(loc=0, scale=1, size=discriminator_vector_dim).tolist())
            self.pos_distribution_vector.append(np.random.normal(loc=0, scale=1, size=generator_vector_dim).tolist())
        self.neg_distribution_vector = torch.FloatTensor(self.neg_distribution_vector).to(device)
        self.pos_distribution_vector = torch.FloatTensor(self.pos_distribution_vector).to(device)