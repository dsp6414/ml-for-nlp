import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import pdb

USE_CUDA = True if torch.cuda.is_available() else False

def loss_func(recon_x, x, mu, logvar, img_sz):
    criterion = F.binary_cross_entropy(recon_x, x.view(-1, img_sz), size_average=False)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL closed form

    return criterion + kl_div

def train(model, train_loader, epoch, optimizer):
    model.train()
    total_loss = 0
    for batch_id, (img, label) in enumerate(train_loader):
        img = Variable(img)
        if USE_CUDA:
            img = img.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = loss_func(recon_batch, img, mu, logvar, img.size()[2]*img.size()[3])
        loss.backward()
        total_loss += loss.data[0]
        optimizer.step()

        if batch_id % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(img), len(train_loader.dataset),
                100. * batch_id / len(train_loader),
                loss.data[0] / len(img)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, total_loss / len(train_loader.dataset)))

def train_minimax(discriminator_model, generative_model, train_loader, epoch, D_optimizer, G_optimizer, batch_size):
    criterion = nn.BCELoss()
    discriminator_model.train()
    generative_model.train()

    d_avg_loss = 0
    g_loss = 0
    number_generator_obs = 0
    number_discriminator_obs = 0

    for batch_id, (img, label) in enumerate(train_loader):
        img = Variable(img)
        if USE_CUDA:
            img = img.cuda()

        # TRAINING DISCRMINATOR
        D_optimizer.zero_grad()

        ## FAKE DATA
        # Generate some noise from normal dist
        z = torch.randn((batch_size/2, 1)) # [batch_size x g_input_dim]
        z = Variable(z)
        if USE_CUDA: 
            z = z.cuda()

        # Pass into generator
        fake_img = generative_model(z).detach() # is .detach() necessary?
        fake_decision = discriminator_model(fake_img)
        desired_fake_decision = Variable(torch.zeros((batch_size/2, 1)))
        if USE_CUDA:
            desired_fake_decision = desired_fake_decision.cuda()
        fake_loss = criterion(fake_decision, desired_fake_decision)
        fake_loss.backward()

        ## REAL DATA:
        real_img = img.view(batch_size, -1)
        real_decision = discriminator_model(real_img) # batch_size x 1
        desired_real_decision= Variable(torch.ones((batch_size/2,1)))
        if USE_CUDA:
            desired_real_decision = desired_real_decision.cuda()
        real_loss = criterion(real_decision, desired_real_decision)
        real_loss.backward()
        d_avg_loss += .5 * (fake_loss.data[0] + real_loss.data[0])
        D_optimizer.step()

        number_discriminator_obs += batch_size/2

        ## TRAINING GENERATOR
        G_optimizer.zero_grad()
        # Some noise as input
        generator_batch_size = 1
        z = torch.randn((generator_batch_size, 1)) # [batch_size x g_input_dim]
        z = Variable(z)
        if USE_CUDA: 
            z = z.cuda()

        fake_imgs = generative_model(z)
        desired_genuine = Variable(torch.ones(generator_batch_size, 1))
        if USE_CUDA:
            desired_genuine = desired_genuine.cuda()
        discriminator_output = discriminator_model(fake_imgs)
        gen_loss = criterion(discriminator_output, desired_genuine)
        gen_loss.backward()
        G_optimizer.step()

        g_loss += gen_loss.data[0]
        number_generator_obs += generator_batch_size

        if batch_id % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Generator Loss: {:.6f}, Disciminator Loss: {:.6f}'.format(
                epoch, batch_id * len(img), len(train_loader.dataset),
                100. * batch_id / len(train_loader),
                gen_loss.data[0] / float(generator_batch_size), .5 * (fake_loss.data[0] + real_loss.data[0]) / float(number_discriminator_obs)))
    print('====> Epoch: {} Generator loss: {:.4f}, Discr. Loss: {:.4f}'.format(
          epoch, g_loss/ number_generator_obs, d_avg_loss / number_discriminator_obs))




def eval(model, data_loader, epoch, batch_sz=100): # maybe need to pass epoch
    model.eval()
    total_loss = 0
    for i, (img, label) in enumerate(data_loader):
        if USE_CUDA:
            img = img.cuda()
        img = Variable(img, volatile=True)
        recon_batch, mu, logvar = model(img)
        img_width = img.size()[2]
        img_height = img.size()[3]
        total_loss += loss_func(recon_batch, img, mu, logvar, img_width * img_height).data[0]

        if i == 0:
            n = min(img.size(0), 8)

            comparison = torch.cat([img[:n],
                                  recon_batch.view(batch_sz, 1, img_width, img_height)[:n]]) #batch_sz is the first one
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    total_loss /= len(data_loader.dataset)
    print('====> Eval set loss: {:.4f}'.format(total_loss))
