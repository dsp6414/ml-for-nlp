from torchtext import data
from torchtext import datasets
import spacy

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

BOS_WORD = '<s>'
EOS_WORD = '</s>'
MAX_LEN = 20
BATCH_SIZE = 32
EPOCHS = 1

DE = data.Field(tokenize=tokenize_de)
EN = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
										 filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
										 len(vars(x)['trg']) <= MAX_LEN)
print(train.fields)
print(len(train))
print(vars(train[0]))

MIN_FREQ = 5
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)
print(DE.vocab.freqs.most_common(10))
print("Size of German vocab", len(DE.vocab))
print(EN.vocab.freqs.most_common(10))
print("Size of English vocab", len(EN.vocab))
print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"]) #vocab index for <s>, </s>

train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
												  repeat=False, sort_key=lambda x: len(x.src))

batch = next(iter(train_iter))
print("Source")
print(batch.src)
print("Target")
print(batch.trg)

# Fix these!!
#encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
#decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

for epoch in range(EPOCHS):
    loss = utils.train(x, y, encoder, decoder, encoder_optm, decoder_optm, criterion)
    print(str(epoch) + "EPOCH LOSS: " + str(loss))

    plot_losses += loss

    if epoch % 5  == 0:
        plot_loss_avg = plot_losses / 5.
        plot_losses.append(plot_loss_avg)
        plot_losses = 0



