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

def train_minimax(discriminator_model, generative_model, train_loader, epoch, D_optimizer, G_optimizer, d_steps, g_steps, batch_size):
    criterion = nn.BCELoss()
    discriminator_model.train()
    generative_model.train()

    epoch_d_loss = 0
    epoch_g_loss = 0
    number_generator_obs = 0
    number_discriminator_obs = 0

    def train_generator():
        ## TRAINING GENERATOR ################################
        G_optimizer.zero_grad()
        # Some noise as input
        z = torch.randn(batch_size, 1) # [batch_size x g_input_dim]
        z = Variable(z)
        if USE_CUDA: 
            z = z.cuda()

        fake_imgs = generative_model(z)
        desired_genuine = Variable(torch.ones(batch_size, 1))
        if USE_CUDA:
            desired_genuine = desired_genuine.cuda()
        discriminator_output = discriminator_model(fake_imgs)
        gen_loss = criterion(discriminator_output, desired_genuine)
        gen_loss.backward()
        G_optimizer.step()

        batch_g_loss = gen_loss.data[0]
        return batch_g_loss, batch_size


    for batch_id, (img, label) in enumerate(train_loader):
        #### DONE TRAINING DISCRIMINATOR -> TRAIN GENERATOR FOR G_STEPS
        if batch_id % d_steps == 0:
            for i in range(g_steps):
                batch_g_loss, num_obs = train_generator()
                number_generator_obs += 1
                epoch_g_loss += batch_g_loss

        ## CONTINUE TRAINING DISCRIMINATOR
        img = Variable(img)
        if USE_CUDA:
            img = img.cuda()

        ##  TRAINING DISCRMINATOR #####################################
        D_optimizer.zero_grad()

        ## FAKE DATA
        # Generate some noise from normal dist
        z = torch.randn(batch_size, 1) # [batch_size x g_input_dim]
        z = Variable(z)
        if USE_CUDA: 
            z = z.cuda()


        if batch_id % 2 == 0:
            # Pass into generator
            fake_img = generative_model(z).detach() # is .detach() necessary?
            fake_decision = discriminator_model(fake_img)
            desired_fake_decision = Variable(torch.zeros((batch_size, 1)))
            if USE_CUDA:
                desired_fake_decision = desired_fake_decision.cuda()
            fake_loss = criterion(fake_decision, desired_fake_decision)
            d_batch_loss = fake_loss.data[0]
            fake_loss.backward()
            
        else:
            ## REAL DATA:
            real_img = img.view(batch_size, -1)
            real_decision = discriminator_model(real_img) # batch_size x 1
            desired_real_decision= Variable(torch.ones((batch_size,1)))
            if USE_CUDA:
                desired_real_decision = desired_real_decision.cuda()
            real_loss = criterion(real_decision, desired_real_decision)
            # d_avg_loss += .5 * (fake_loss.data[0] + real_loss.data[0])
            d_batch_loss = real_loss.data[0]
            real_loss.backward()

        if d_batch_loss > 0.5:
            print("STEP", d_batch_loss)
            D_optimizer.step()

        epoch_d_loss += d_batch_loss
        number_discriminator_obs += 1

        if batch_id % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Generator Loss: {:.6f}, Disciminator Loss: {:.6f}'.format(
                epoch, batch_id * len(img), len(train_loader.dataset),
                100. * batch_id / len(train_loader),
                batch_g_loss, d_batch_loss))
    print('====> Epoch: {} Generator loss: {:.4f}, Discr. Loss: {:.4f}'.format(
          epoch, epoch_g_loss/ number_generator_obs, epoch_d_loss / number_discriminator_obs))

def eval_minimax(discriminator_model, generative_model, data_loader, epoch, batch_size):
    criterion = nn.BCELoss()
    discriminator_model.eval()
    generative_model.eval()

    def eval_generator(batch_size):
        # Some noise as input
        z = torch.randn(batch_size, 1) # [batch_size x g_input_dim]
        z = Variable(z)
        if USE_CUDA: 
            z = z.cuda()

        fake_imgs = generative_model(z)
        desired_genuine = Variable(torch.ones(batch_size, 1))
        if USE_CUDA:
            desired_genuine = desired_genuine.cuda()
        discriminator_output = discriminator_model(fake_imgs)
        gen_loss = criterion(discriminator_output, desired_genuine)

        batch_g_loss = gen_loss.data[0]
        return batch_g_loss, batch_size

    epoch_d_loss = 0.0
    epoch_g_loss = 0.0

    n_batches = 0

    # REAL DATA: DISCRIMINATOR
    for batch_id, (img, label) in enumerate(data_loader): 
        n_batches += 1
        img = Variable(img)
        if USE_CUDA:
            img = img.cuda()

        real_img = img.view(batch_size, -1)
        real_decision = discriminator_model(real_img) # batch_size x 1
        desired_real_decision= Variable(torch.ones((batch_size,1)))
        if USE_CUDA:
            desired_real_decision = desired_real_decision.cuda()
        real_loss = criterion(real_decision, desired_real_decision)
        # d_avg_loss += .5 * (fake_loss.data[0] + real_loss.data[0])
        epoch_d_loss += real_loss.data[0]

    # FAKE DATA: DISCRIMINATOR
    n_obs = n_batches * batch_size

    ## FAKE DATA
    # Generate some noise from normal dist
    z = torch.randn(n_obs, 1) # [batch_size x g_input_dim]
    z = Variable(z)
    if USE_CUDA: 
        z = z.cuda()
    # Pass into generator
    fake_img = generative_model(z).detach() # is .detach() necessary?
    fake_decision = discriminator_model(fake_img)
    desired_fake_decision = Variable(torch.zeros((n_obs, 1)))
    if USE_CUDA:
        desired_fake_decision = desired_fake_decision.cuda()
    fake_loss = criterion(fake_decision, desired_fake_decision)
    d_batch_loss = fake_loss.data[0]

    ## AVERAGE REAL AND FAKE
    epoch_d_loss = epoch_d_loss *.5 + d_batch_loss *.5

    ## GENERATOR:
    epoch_g_loss, num_obs = eval_generator(n_obs)

    print('====> Eval set loss: Discriminator: {:.4f}, Generator: {:.4f}'.format(epoch_d_loss, epoch_g_loss))





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
