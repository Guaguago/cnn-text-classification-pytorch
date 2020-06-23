#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
# import train
import mydatasets
import torch.autograd as autograd

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=True, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# load SST dataset
def sst(text_field, label_field, **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(args.batch_size,
                     len(dev_data),
                     len(test_data)),
        **kargs)
    return train_iter, dev_iter, test_iter


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(args.batch_size, len(dev_data)),
        **kargs)
    return train_iter, dev_iter


# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(args)
snapshot = '/Users/xuchen/core/pycharm/project/cnn-text-classification-pytorch/snapshot/best_steps_11513.pt'
if snapshot is not None:
    print('\nLoading model from {}...'.format(snapshot))
    cnn.load_state_dict(torch.load(snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()


def sent_acc(samples, model, text_field, cuda_flag, positive=True, ):
    size = len(samples)
    model.eval()
    # text = text_field.tokenize(text)
    outputs = torch.tensor([], dtype=torch.int64)
    for sample in samples:
        sample = text_field.preprocess(sample)
        sample = [[text_field.vocab.stoi[x] for x in sample]]
        # inputs.append(sample)
        x = torch.tensor(sample)
        x = autograd.Variable(x)
        if cuda_flag:
            x = x.cuda()
        # print(x)
        output = model(x)
        _, predicted = torch.max(output, 1)  # logits
        outputs = torch.cat([outputs, predicted])

    target = [1] * size if positive else [0] * size
    target = torch.tensor(target)
    corrects = outputs == target
    corrects = corrects.sum()
    accuracy = 100.0 * corrects / size
    # return label_feild.vocab.itos[predicted.data[0][0]+1]
    return accuracy

def calculate_acc(file_pos, file_neg, label):
    # mean_acc, pos_acc, neg_acc = None, None, None
    file_pos = '{}/{}'.format(file_pos, label)
    with open(file_pos, 'r') as f:
        samples = f.read().split('<|endoftext|>')
        samples = [s for s in samples if len(s.split()) > 20]

        # samples = ['I love you so much !', 'So cool good . happy birthday ! ']
        pos_acc = sent_acc(samples, cnn, text_field, args.cuda, positive=True)
        # print('{} = {}'.format(l, pos_acc))

    file_neg = '{}/{}'.format(file_neg, label)
    with open(file_neg, 'r') as f:
        samples = f.read().split('<|endoftext|>')
        samples = [s for s in samples if len(s.split()) > 20]

        # samples = ['I love you so much !', 'So cool good . happy birthday ! ']
        neg_acc = sent_acc(samples, cnn, text_field, args.cuda, positive=False)
        # print('{} = {}'.format(l, neg_acc))
    mean_acc = (pos_acc + neg_acc) / 2
    return mean_acc, pos_acc, neg_acc


# train or predict
label = ['B', 'BR', 'BC', 'BCR']
file_pos = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/vad_abs/positive'
file_neg = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/vad_abs/negative'
# file_pos = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/pplm/reversed/positive'
# file_neg = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/pplm/reversed/negative'
mean_acc, pos_acc, neg_acc = calculate_acc(file_pos, file_neg, 'BC')
print(pos_acc)
print(neg_acc)
print(mean_acc.item())

# for l in label:
#     pos_acc, neg_acc = None, None
#     with open('/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/positive/{}'.format(l), 'r') as f:
#         samples = f.read().split('<|endoftext|>')
#         samples = [s for s in samples if len(s.split()) > 20]
#
#         # samples = ['I love you so much !', 'So cool good . happy birthday ! ']
#         pos_acc = sent_acc(samples, cnn, text_field, args.cuda, positive=True)
#         # print('{} = {}'.format(l, pos_acc))
#
#     with open('/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/negative/{}'.format(l), 'r') as f:
#         samples = f.read().split('<|endoftext|>')
#         samples = [s for s in samples if len(s.split()) > 20]
#
#         # samples = ['I love you so much !', 'So cool good . happy birthday ! ']
#         neg_acc = sent_acc(samples, cnn, text_field, args.cuda, positive=False)
#         # print('{} = {}'.format(l, neg_acc))
#
#     print('{} = {}'.format(l, (pos_acc + neg_acc) / 2))


