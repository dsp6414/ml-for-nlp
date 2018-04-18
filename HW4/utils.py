import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import math
import pdb

USE_CUDA = True if torch.cuda.is_available() else False

batch_size = 128

def loss_func(recon_x, x, mu, logvar, img_sz):
    criterion = F.binary_cross_entropy(recon_x, x.view(-1, img_sz), size_average=False)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL closed form
    return criterion + kl_div

    # recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, img_sz), size_average=False) / 128
    # KLLoss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1. - logvar, 1))
    # return recon_loss + KLLoss

def visualize_model(model, data_loader, batch_sz=128, is_conditional=False):
    f, ax = plt.subplots()
    model.eval()
    for i, (img, label) in enumerate(data_loader):
        if USE_CUDA:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img, volatile=True) #volatile: uses minimal memory, requires_grad = False
        label = Variable(label.float(), volatile=True)

        if is_conditional:
            recon_batch, mu, logvar = model(img, label)
        else:
            recon_batch, mu, logvar = model(img)

        x = mu[:, 0]
        y = mu[:, 1]

        colors = label.data.cpu().numpy()
        ax.scatter(x, y, c=colors)
        f.savefig('scatter_'+str(i)+'.png')

    f.savefig('scatterplot.png',)


def train(model, train_loader, epoch, optimizer, noise=False):
    model.train()
    total_loss = 0

    noise_factor = 0.25
    for batch_id, (img, label) in enumerate(train_loader):
        img = Variable(img)
        label = Variable(label.float())
        if USE_CUDA:
            img = img.cuda()
            label = label.cuda()
        optimizer.zero_grad()

        if noise:
            img_noise = img + noise_factor * Variable(torch.randn(img.size()))
            img_noise.data.clamp(0., 1.)
            recon_batch, mu, logvar = model(img_noise, label)
        else:
            recon_batch, mu, logvar = model(img)
        loss = loss_func(recon_batch, img, mu, logvar, img.size()[2]*img.size()[3])
        loss.backward()
        total_loss += loss.data[0]
        optimizer.step()

        # if batch_id % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_id * len(img), len(train_loader.dataset),
        #         100. * batch_id / len(train_loader),
        #         loss.data[0] / len(img)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, total_loss / len(train_loader.dataset)))

def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels, d_optimizer, criterion):
    discriminator.zero_grad()
    outputs = discriminator(images)
    real_loss = criterion(outputs, real_labels)
    real_score = outputs
    
    outputs = discriminator(fake_images) 
    fake_loss = criterion(outputs, fake_labels)
    fake_score = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward(retain_graph=True)
    d_optimizer.step()
    return d_loss.data[0], real_score, fake_score

def train_generator(generator, discriminator_outputs, real_labels, g_optimizer, criterion):
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data[0]

def train_minimax(discriminator_model, generative_model, train_loader, epoch, D_optimizer, G_optimizer, d_steps, g_steps, batch_size, noise_dim):
    generator = generative_model
    discriminator = discriminator_model

    # draw samples from the input distribution to inspect the generation on training 
    num_test_samples = 16
    test_noise = Variable(torch.randn(num_test_samples, noise_dim).cuda())

    criterion = nn.BCELoss()
    discriminator_model.train()
    generative_model.train()

    epoch_d_loss = 0
    epoch_g_loss = 0
    number_generator_obs = 0
    number_discriminator_obs = 0

    num_batches = len(train_loader)

    for n, (images, _) in enumerate(train_loader):
        images = Variable(images.cuda()).view(batch_size, -1)
        real_labels = Variable(torch.ones(images.size(0)).cuda())
        
        # Sample from generator
        noise = Variable(torch.randn(images.size(0), noise_dim).cuda())
        fake_images = generator(noise)
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())
        
        # Train the discriminator
        d_loss, real_score, fake_score = train_discriminator(discriminator, images, real_labels, fake_images, fake_labels, D_optimizer, criterion)
        
        # Sample again from the generator and get output from discriminator
        noise = Variable(torch.randn(images.size(0), noise_dim).cuda())
        fake_images = generator(noise)
        
        outputs = discriminator(fake_images)

        # Train the generator
        g_loss = train_generator(generator, outputs, real_labels, G_optimizer, criterion)
        
        if (n+1) % 100 == 0:
            test_images = generator(test_noise)

            test_images = test_images.view(num_test_samples, 1, 28, 28)

            for num, fake_img in enumerate(test_images):
                save_image(fake_img.data,
                         'results_gan/generated_epoch_' + str(epoch) + '_ex'+str(num) +'.png', nrow=28, padding=0)

            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
                  'D(x): %.2f, D(G(z)): %.2f' 
                  %(epoch + 1, epoch, n+1, num_batches, d_loss, g_loss,
                    real_score.data.mean(), fake_score.data.mean()))

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

        img_width = img.size()[2]
        img_height = img.size()[3]

        real_img = img.view(batch_size, -1)
        real_decision = discriminator_model(real_img) # batch_size x 1
        desired_real_decision= Variable(torch.ones((batch_size,1)))
        if USE_CUDA:
            desired_real_decision = desired_real_decision.cuda()
        real_loss = criterion(real_decision, desired_real_decision)
        # d_avg_loss += .5 * (fake_loss.data[0] + real_loss.data[0])
        epoch_d_loss += real_loss.data[0]

    epoch_d_loss = epoch_d_loss / n_batches

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
    print(num_obs)

    n = min(8, fake_img.size(0))

    fake_img = fake_img.view(batch_size * n_batches, 1, img_width, img_height)[0]
    save_image(fake_img.data,
                     'results_gan/generated_' + str(epoch) + '.png', nrow=img_height, padding=0)


    print('====> Eval set loss: Generator Loss: {:.4f}, Discr.: {:.4f}'.format(epoch_g_loss, epoch_d_loss))


def gen_interpolated_examples(model, noise_dim, model_name, use_decoder=False):
    model.eval()

    noise_1 = Variable(torch.randn(1, noise_dim)) # [batch_size x g_input_dim]
    noise_2 = Variable(torch.randn(1, noise_dim))
    if USE_CUDA: 
        noise_1 = noise_1.cuda()
        noise_2 = noise_2.cuda()

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    new_noises = [alpha * noise_1 + (1 - alpha) * noise_2 for alpha in alphas]

    for alpha, noise in zip(alphas, new_noises):
        fake_img = model.decode(noise) if use_decoder else model(noise)
        fake_img = fake_img.view(1, 1, 28, 28)

        save_image(fake_img.data,
                 'results_interp/' + model_name + str(alpha).replace('.','') + '.png', nrow=28, padding=0)




def eval(model, data_loader, epoch, batch_sz=128, is_conditional=False): # maybe need to pass epoch
    model.eval()
    total_loss = 0
    for i, (img, label) in enumerate(data_loader):
        if USE_CUDA:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img, volatile=True) #volatile: uses minimal memory, requires_grad = False
        label = Variable(label.float(), volatile=True)

        if is_conditional:
            recon_batch, mu, logvar = model(img, label)
        else:
            recon_batch, mu, logvar = model(img)
        img_width = img.size()[2]
        img_height = img.size()[3]
        total_loss += loss_func(recon_batch, img, mu, logvar, img_width * img_height).data[0]

        if i == 0 and epoch % 10 == 0:
            n = min(img.size(0), 8)
            comparison = torch.cat([img[:n],
                                  recon_batch.view(batch_sz, 1, img_width, img_height)[:n]]) #batch_sz is the first one
            save_image(comparison.data.cpu(),
                     'results/reconstruction_meh_' + str(epoch) + '.png', nrow=n)

    total_loss /= len(data_loader.dataset)
    print('====> Eval set loss: {:.4f}'.format(total_loss))


