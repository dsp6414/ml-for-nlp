import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors, GloVe
import pdb

torch.manual_seed(1)

def get_batch(batch):
    text = batch.text[:-1,:]
    target = batch.text[1:,:].view(-1)
    if torch.cuda.is_available():
        text = text.cuda()
        target = target.cuda()
    return text, target

def reset_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(reset_hidden(v) for v in h)

def train_batch(model, text, target, hidden, criterion, optimizer, grad_norm):
    # initialize hidden vectors
    hidden = reset_hidden(hidden) # This includes (hidden, cell)
    model.zero_grad()
    output, hidden = model(text, hidden) # output: [bptt_len-1 x batch x vocab_size]
    output_flat = output.view(-1, model.vocab_size)
    loss = criterion(output_flat, target) # target: [bptt_len-1 x batch]
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), max_norm=grad_norm)
    optimizer.step()
    return loss.data[0], hidden

def train(model, train_iter, num_epochs, criterion, optimizer, scheduler=None, grad_norm=5):
    model.train()
    filename = 'lstm_large_hidden'
    hidden = model.init_hidden()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_iter:
            text, target = get_batch(batch)
            batch_loss, hidden = train_batch(model, text, target, hidden, criterion, optimizer, grad_norm)
            total_loss += batch_loss
        if scheduler:
            scheduler.step()
            print("learning rate: " + str(scheduler.get_lr()))
        print("Epoch " + str(epoch) + " Loss: " + str(total_loss))
        if epoch % 5 == 0:
            print("SAVING MODEL #" + str(epoch))
            torch.save(model.state_dict(), filename + str(epoch) + ".sav")

def evaluate(model, iter_data, criterion):
    model.eval()
    total_loss = 0.0
    total_len = 0.0
    h = model.init_hidden()
    for batch in iter_data:
        text, target = get_batch(batch)
        probs, h = model(text, h)
        probs_flat = probs.view(-1, model.vocab_size)
        total_loss += len(text) * criterion(probs_flat, target).data
        total_len += len(text)
        h = reset_hidden(h)
    print("Total Loss ", total_loss[0])
    print("Total Len ", total_len)
    print(total_loss[0] / total_len)
    return total_loss[0] / total_len
