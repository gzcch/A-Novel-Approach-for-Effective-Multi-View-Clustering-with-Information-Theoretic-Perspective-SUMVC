from __future__ import print_function, division
import argparse
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear

from sklearn import preprocessing
import warnings
from time import time
warnings.filterwarnings('ignore')
min_max_scaler = preprocessing.MinMaxScaler()
normalize = preprocessing.Normalizer()

min_max_scaler = preprocessing.MinMaxScaler()
normalize = preprocessing.Normalizer()

from utils import cluster_acc, WKLDiv, multiViewDataset2,imagedataset
import os

cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


"""class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, n_z):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z),requires_grad=True)

    def forward(self, x):
        q = 1.0 / (1 + torch.sum(
            torch.pow(x.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        return q
"""

class SingleViewModel(nn.Module):

    def __init__(self,
                 n_input, n_z, n_clusters, pretrain):
        super(SingleViewModel, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input[0], 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),

        )
        self.latentmu = nn.Sequential(
            nn.Linear(64 * 4 * 4, n_z),
            nn.ReLU()

        )
        self.latentep = nn.Sequential(
            nn.Linear(64 * 4 * 4, n_z),
            nn.ReLU()

        )
        self.delatent = nn.Sequential(
            nn.Linear(n_z, 64 * 4 * 4),
            nn.ReLU()
        )
        for m in self.encoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_input[0], (4, 4), stride=2, padding=1),
            nn.Sigmoid()
        )

        for m in self.decoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)


        for m in self.decoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)

        self.pretrain = pretrain



    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        z = mu + eps * std
        return z


    def forward(self, x):
        mu =self.latentmu(self.encoder(x).view(x.shape[0],-1))
        log = self.latentep(self.encoder(x).view(x.shape[0],-1))
        z = self.reparameterize(mu, log)
        #y=self.classifier(z)

        # z = torch.nn.functional.normalize(z, p=1.0, dim=1)


        
        x_bar=self.decoder(self.delatent(z).view(-1,64,4,4))
        #q = self.clusteringLayer(z)

        return x_bar, z, mu, log


class MultiViewModel(nn.Module):

    def __init__(self,
                 n_input,
                 n_z,
                 n_clusters,
                 pretrain,
                 save_path):
        super(MultiViewModel, self).__init__()
        self.pretrain = pretrain
        self.save_path = save_path
        self.n_clusters = n_clusters
        self.viewNumber = args.viewNumber

        classifier = list()

        for viewIndex in range(self.viewNumber):
            classifier.append(
                nn.Sequential(
                    Linear(args.viewNumber * n_z, n_clusters),
                    nn.Softmax(dim=1)
                )
            )
        self.classifier = nn.ModuleList(classifier)

        if args.share:
            self.aes = SingleViewModel(
                n_input=n_input[0],
                n_z=n_z,
                n_clusters=self.n_clusters,
                pretrain=self.pretrain)


        else:

            aes = []

            for viewIndex in range(self.viewNumber):
                aes.append(SingleViewModel(
                n_input=n_input[0],
                n_z=n_z,
                n_clusters=self.n_clusters,
                pretrain=self.pretrain))


            self.aes = nn.ModuleList(aes)

    def forward(self, x):
        outputs = []
        if args.share:
            for viewIndex in range(self.viewNumber):
                outputs.append(self.aes(x[viewIndex]))
        else:
            for viewIndex in range(self.viewNumber):
                outputs.append(self.aes[viewIndex](x[viewIndex]))

        return outputs


class Gaussian(nn.Module):
 
    def __init__(self, num_classes, latent_dim):
        super(Gaussian, self).__init__()
        self.num_classes = num_classes


        self.mean = nn.Parameter(torch.zeros(self.num_classes, latent_dim))

    def forward(self, z):
        z = z.unsqueeze(1)
        return z - self.mean.unsqueeze(0)

#    def compute_output_shape(self, input_shape):
#        return (None, self.num_classes, input_shape[-1])
class Multi_Gaussian(nn.Module):

    def __init__(self, num_classes, latent_dim, viewNumber):
        super(Multi_Gaussian, self).__init__()
        self.num_classes = num_classes
        self.viewNumber = viewNumber
        self.latent_dim = latent_dim

        gus_list = list()
        for viewIndex in range(self.viewNumber):
            gus_list.append(Gaussian(self.num_classes, self.viewNumber*self.latent_dim).cuda())
        self.gus_list = nn.ModuleList(gus_list)

    def forward(self, z, index):

        return self.gus_list[index](z)

def make_qp(x, centroids):
    q = 1.0 / (1.0*0.001 + torch.sum(torch.pow(x.unsqueeze(1) - torch.tensor(centroids).cuda(), 2), 2))
    q = (q.t() / torch.sum(q, 1)).t()
    p = target_distribution(q)
    return q, p


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()




 
def pretrain_aes():
    save_path = args.save_path
    viewNumber = args.viewNumber
    model = MultiViewModel(
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        pretrain=True,
        save_path=args.save_path).cuda()
    #if args.noise==0:
    #model.load_state_dict(torch.load(args.save_path))
    dataset = imagedataset(args.dataset, args.viewNumber, args.method, True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)
    #gaussian=Gaussian(args.n_clusters, args.n_z).cuda()
    multi_gaussian = Multi_Gaussian(args.n_clusters, args.n_z, args.viewNumber).cuda()

    gamma_1 = args.gamma_1
    for epoch in range(1000):
        for batch_idx, (x, _, _) in enumerate(dataLoader):
            loss=0.
            mseloss=0.
            kl_loss=0.
            cat_loss=0.
            for viewIndex in range(viewNumber):
                x[viewIndex] = x[viewIndex].cuda()        
            output = model(x)

            for viewIndex in range(viewNumber):
                if viewIndex == 0:
                    z_all = output[viewIndex][1]
                else:
                    z_all = torch.cat((z_all, output[viewIndex][1]), dim=1)
            for viewIndex in range(viewNumber):
                z_prior_mean=multi_gaussian(z_all,viewIndex)
                mseloss += loss + F.mse_loss(output[viewIndex][0], x[viewIndex])
                #  n*c*1 , n*c*d ->  n*c*d
                temp = -0.5 * (output[viewIndex][3].unsqueeze(1) -
                               torch.sum(torch.square(z_prior_mean),dim=2,keepdim=True)/((args.n_z)))
                y_v = model.classifier[viewIndex](z_all)
                kl_loss += torch.sum(torch.mean(y_v.unsqueeze(-1) * temp), dim=0)
                cat_loss += -torch.sum(torch.mean(y_v * torch.log(y_v + 1e-8), dim=0))

                #loss = loss +  nn.KLDivLoss(reduction='mean')(torch.tensor(0.001).log(), output[viewIndex][1])
            loss =mseloss + gamma_1 * kl_loss + 0*cat_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.print:
                if epoch % 10==0 and batch_idx % 50== 0:
                    print('epoch:{},batch:{},mseloss:{:.4f}, kl_loss{:.4f}, loss{:.4f}'.format(epoch,batch_idx, mseloss,kl_loss, loss))

    torch.save(model.state_dict(), args.save_path+str(args.share)+str('.pkl'))
    torch.save(multi_gaussian.state_dict(),args.save_path+str(args.share)+str('_multigaussian')+str('.pkl'))
    dataLoader = DataLoader(dataset, batch_size=args.instanceNumber, shuffle=False)
    for batch_idx, (x, y, _) in enumerate(dataLoader):
        for viewIndex in range(viewNumber):
            x[viewIndex] = x[viewIndex].cuda()
        output = model(x)
        y = y.data.cpu().numpy()
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)

    for viewIndex in range(args.viewNumber):
        z_v = output[viewIndex][1]

        if (viewIndex) == 0:
            z_all = z_v

        else:
            z_all = torch.cat((z_all, z_v), dim=1)

    kmeans.fit_predict(z_all.cpu().detach().data.numpy())

    y_pred = kmeans.labels_
    # y_pred = (np.argmax(qpred.cpu().detach().data.numpy(), axis=1))
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('z_all :acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

    y_pred = 0
    for viewIndex in range(viewNumber):
        y_pred += model.classifier[viewIndex](z_all)


    y_pred = np.argmax(y_pred.cpu().detach().data.numpy(), axis=1)
    # y_pred = (np.argmax(qpred.cpu().detach().data.numpy(), axis=1))
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('y :acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))



def MMD_loss(x, y, sigma=1.0):
    # Compute the RBF kernel matrix for x and y
    Kxx = compute_rbf_kernel(x, x, sigma)
    Kxy = compute_rbf_kernel(x, y, sigma)
    Kyy = compute_rbf_kernel(y, y, sigma)

    loss = torch.sum(Kxx) + torch.sum(Kyy) - 2 * torch.sum(Kxy)

    return loss/(args.batch_size*args.batch_size)

def compute_rbf_kernel(x, y, sigma=1.0):
    """
    Computes the RBF kernel matrix between two sets of embeddings x and y.

    Args:
        x: A tensor of shape (batch_size_1, embedding_dim).
        y: A tensor of shape (batch_size_2, embedding_dim).
        sigma: The bandwidth of the RBF kernel.

    Returns:
        A tensor of shape (batch_size_1, batch_size_2) containing the RBF kernel matrix.
    """

    # Compute the squared Euclidean distance matrix between x and y
    dist = torch.sum(x ** 2, dim=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
    # Compute the RBF kernel matrix
    kernel = torch.exp(-dist / (2 * sigma ** 2))

    return kernel
def gaussian_kl_divergence(mu1, logvar1, mu2, logvar2):
    """
    Calculate the KL divergence between two Gaussian distributions

    Args:
        mu1: tensor, mean of the first Gaussian distribution
        logvar1: tensor, log variance of the first Gaussian distribution
        mu2: tensor, mean of the second Gaussian distribution
        logvar2: tensor, log variance of the second Gaussian distribution

    Returns:
        kl_divergence: tensor, KL divergence between two Gaussian distributions
    """
    # Calculate the diagonal elements of covariance matrix
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    # Calculate the KL divergence
    kl_divergence = 0.5 * (torch.sum(var1 / var2, dim=-1)
                           + torch.sum((mu2 - mu1).pow(2) / var2, dim=-1)
                           + torch.sum(logvar2, dim=-1)
                           - torch.sum(logvar1, dim=-1)
                           - mu1.shape[-1])

    return torch.sum(kl_divergence)/(mu1.shape[0]*mu1.shape[1])


"""
   def gaussian_dskl_divergence(mu1, logvar1, mu2, logvar2):

       Calculate the DSKL divergence between two Gaussian distributions

       Args:
           mu1: tensor, mean of the first Gaussian distribution
           logvar1: tensor, log variance of the first Gaussian distribution
           mu2: tensor, mean of the second Gaussian distribution
           logvar2: tensor, log variance of the second Gaussian distribution

       Returns:
           dskl_divergence: tensor, DSKL divergence between two Gaussian distributions

       kl_divergence1 = gaussian_kl_divergence(mu1, logvar1, mu2, logvar2)
       kl_divergence2 = gaussian_kl_divergence(mu2, logvar2, mu1, logvar1)
       dskl_divergence = 0.5 * (kl_divergence1 + kl_divergence2)
       return dskl_divergence
   """

def fineTuning():
    model = MultiViewModel(
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        pretrain=True,
        save_path=args.save_path).cuda()

    viewNumber=args.viewNumber
    multi_gaussian = Multi_Gaussian(args.n_clusters, args.n_z, args.viewNumber).cuda()
    model.load_state_dict(torch.load(args.save_path+str(args.share)+str('.pkl')))
    multi_gaussian.load_state_dict(torch.load(args.save_path+str(args.share)+str('_multigaussian')+str('.pkl')))
    dataset = imagedataset(args.dataset, args.viewNumber, args.method, True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # intalize graph gf,z_all
    gamma_1 = args.gamma_1
    gamma_2 = args.gamma_2
    gamma_3 = args.gamma_3
    gamma_4=  args.gamma_4

    for epoch in tqdm(range(1000)):
        for batch_idx, (x, _, _) in enumerate(dataLoader):
            loss=0.
            mseloss=0.
            kl_loss=0.
            dskl_loss=0.
            mmd_loss=0.
            cat_loss=0.
            for viewIndex in range(viewNumber):
                x[viewIndex] = x[viewIndex].cuda()        
            output = model(x)


            for viewIndex in range(viewNumber):
                if viewIndex == 0:
                    z_all = output[viewIndex][1]
                else:
                    z_all = torch.cat((z_all, output[viewIndex][1]), dim=1)
            for viewIndex in range(viewNumber):
                z_prior_mean=multi_gaussian(z_all,viewIndex)
                mseloss += loss + F.mse_loss(output[viewIndex][0], x[viewIndex])
                temp = -0.5 * (output[viewIndex][3].unsqueeze(1) -
                               torch.sum(torch.square(z_prior_mean),dim=2,keepdim=True)/(args.n_z))
                y_v = model.classifier[viewIndex](z_all)
                kl_loss += torch.sum(torch.mean(y_v.unsqueeze(-1) * temp), dim=0)
                cat_loss += -torch.sum(torch.mean(y_v * torch.log(y_v + 1e-8), dim=0))

            for viewIndex1 in range(0, viewNumber):
                for viewIndex2 in  range(0, viewNumber):
                    if viewIndex1 == viewIndex2: 
                        continue
                    mmd_loss += MMD_loss(output[viewIndex1][1], output[viewIndex2][1])
                    dskl_loss+= gaussian_kl_divergence(output[viewIndex1][2], output[viewIndex1][3], output[viewIndex2][2], output[viewIndex2][3])




            loss = mseloss + gamma_1 * kl_loss + gamma_2 * mmd_loss + gamma_3 * dskl_loss + gamma_4 * cat_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.print :
                if epoch % 50 == 0 and batch_idx% 20==0:
                    print('epoch:{},batch_idx{}'.format(epoch,batch_idx) )
                    print('mseloss:{:.4f}, mmdloss{:.4f}, kl_loss{:.4f}, dsklloss{:.4f},loss{:.4f}'.format(mseloss,mmd_loss, kl_loss,dskl_loss, loss))

    torch.save(model.state_dict(),str(args.dataset)+'gen.pkl')
    torch.save(multi_gaussian.state_dict(),str(args.dataset)+'gen_multigaussian.pkl')


    dataLoader = DataLoader(dataset, batch_size=args.instanceNumber, shuffle=False)

    for batch_idx, (x, y, _) in enumerate(dataLoader):
        for viewIndex in range(viewNumber):
            x[viewIndex] = x[viewIndex].cuda()
        output = model(x)
        y = y.data.cpu().numpy()
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
 
    for viewIndex in range(args.viewNumber):
        z_v = output[viewIndex][1]

        if (viewIndex) == 0:
            z_all = z_v

        else:
            z_all = torch.cat((z_all, z_v), dim=1)


    kmeans.fit_predict(z_all.cpu().detach().data.numpy())

    y_pred = kmeans.labels_
    # y_pred = (np.argmax(qpred.cpu().detach().data.numpy(), axis=1))
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('z_all :acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

    y_pred = 0
    for viewIndex in range(viewNumber):

        y_pred += model.classifier[viewIndex](z_all)


    y_pred = np.argmax(y_pred.cpu().detach().data.numpy(), axis=1)
    # y_pred = (np.argmax(qpred.cpu().detach().data.numpy(), axis=1))
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('y :acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--n_z', default=200, type=int)
    parser.add_argument('--dataset', type=str, default='BDGP')
    parser.add_argument('--share', type=int, default=0)
    #parser.add_argument('--arch', type=int, default=50)
    #parser.add_argument('--gamma', type=int, default=5)
    parser.add_argument('--gamma_1', default=0.1)
    parser.add_argument('--gamma_2', type=int, default=0.1)
    parser.add_argument('--gamma_3', type=int, default=0.1)
    parser.add_argument('--gamma_4',  default=0.1)
    parser.add_argument('--pretrain', default=0)
    #parser.add_argument('--update_interval', default=1000, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    #parser.add_argument('--AR', default=0.95, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    args.dataset = 'MNIST_multi_wzc'
    args.method = 'MNIST_multi_wzc'
    args.share = 1
    args.print = 0


    if args.dataset == 'Multi-COIL-20':
        args.n_input = [[1, 32, 32], [1, 32, 32], [1, 32, 32]]
        args.viewNumber = 3
        args.instanceNumber = 1440
        args.batch_size = 144
        args.n_clusters = 20
        args.save_path = './data/Multi-COIL-20.pkl'

    if args.dataset == 'Multi-Mnist':
        args.n_input = [[1, 32, 32], [1, 32, 32]]
        args.viewNumber = 2
        args.instanceNumber = 10000
        args.batch_size = 2000
        args.n_clusters = 10
        args.save_path = './data/Multi-Mnist.pkl'
        args.n_z = 200

    if args.dataset == 'Multi-FMnist':
        args.n_input = [[1, 32, 32], [1, 32, 32], [1, 32, 32]]
        args.viewNumber = 3
        args.instanceNumber = 10000
        args.batch_size = 2000
        args.n_clusters = 10
        args.save_path = './data/before_Multi-FMnist.pkl'

    if args.dataset == 'Multi-COIL-10':
        args.n_input = [[1, 32, 32], [1, 32, 32], [1, 32, 32]]
        args.viewNumber = 3
        args.instanceNumber = 720
        args.batch_size = 720
        args.n_clusters = 10
        args.save_path = './data/Multi-COIL-10.pkl'
    if args.dataset == 'MNIST_multi_wzc':
        args.n_input = [[1, 32, 32], [1, 32, 32], [1, 32, 32]]
        args.viewNumber = 3
        args.instanceNumber = 1000
        args.batch_size = 1000
        args.n_clusters = 10
        args.save_path = './data/MNIST_multi_wzc.pkl'
        args.n_z = 200

    print(args)
    for args.gamma_1 in [0.1]:
        for args.gamma_2 in [0.1]:
            args.gamma_3 = args.gamma_2
            args.gamma_4 = args.gamma_1
            print('gamma:{}, beta:{}'.format(args.gamma_1, args.gamma_2))
            start =time()

            t0 = time()

            #pretrain_aes()
            fineTuning()

            t1 = time()
            print(t1-t0)
