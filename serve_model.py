import os
import gc
import random
import pprint
from six.moves import range
from markdown2 import markdown
from time import gmtime, strftime
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import options
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from eval_utils.dialog_generate import dialogDump
from eval_utils.rank_answerer import rankABot
from eval_utils.rank_questioner import rankQBot, rankQABots
from utils import utilities as utils
from utils.visualize import VisdomVisualize

from nltk.tokenize import word_tokenize
import numpy as np
import json

# read the command line options
params = options.readCommandLine()

# seed rng for reproducibility
# manualSeed = 1234
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)
# if params['useGPU']:
#     torch.cuda.manual_seed_all(manualSeed)

# setup dataloader
dlparams = params.copy()
dlparams['useIm'] = True
dlparams['useHistory'] = True
dlparams['numRounds'] = 10
splits = ['val']

dataset = VisDialDataset(dlparams, splits)

# Transferring dataset parameters
transfer = ['vocabSize', 'numOptions', 'numRounds']
for key in transfer:
    if hasattr(dataset, key):
        params[key] = getattr(dataset, key)

if 'numRounds' not in params:
    params['numRounds'] = 10

# Always load checkpoint parameters with continue flag
params['continue'] = True

excludeParams = ['batchSize', 'visdomEnv', 'startFrom', 'qstartFrom', 'trainMode', \
    'evalModeList', 'inputImg', 'inputQues', 'inputJson', 'evalTitle', 'beamSize', \
    'enableVisdom', 'visdomServer', 'visdomServerPort']

aBot = None
qBot = None

# load aBot
if params['startFrom']:
    aBot, loadedParams, _ = utils.loadModel(params, 'abot', overwrite=True)
    assert aBot.encoder.vocabSize == dataset.vocabSize, "Vocab size mismatch!"
    for key in loadedParams:
        params[key] = loadedParams[key]
    aBot.eval()

# Retaining certain dataloder parameters
for key in excludeParams:
    params[key] = dlparams[key]

# load qBot
if params['qstartFrom']:
    qBot, loadedParams, _ = utils.loadModel(params, 'qbot', overwrite=True)
    assert qBot.encoder.vocabSize == params[
        'vocabSize'], "Vocab size mismatch!"
    for key in loadedParams:
        params[key] = loadedParams[key]
    qBot.eval()

# Retaining certain dataloder parameters
for key in excludeParams:
    params[key] = dlparams[key]

print("Using split %s" % params['evalSplit'])
dataset.split = params['evalSplit']

print('Loading json file: ' + params['inputJson'])
with open(params['inputJson'], 'r') as fileId:
    info = json.load(fileId)

wordCount = len(info['word2ind'])
# Add <START> and <END> to vocabulary
info['word2ind']['<START>'] = wordCount + 1
info['word2ind']['<END>'] = wordCount + 2
startToken = info['word2ind']['<START>']
endToken = info['word2ind']['<END>']
# Padding token is at index 0
vocabSize = wordCount + 3
print('Vocab size with <START>, <END>: %d' % vocabSize)

# Construct the reverse map
info['ind2word'] = {
    int(ind): word
    for word, ind in info['word2ind'].items()
}

ind_map = lambda words: np.array([info['word2ind'].get(word, info['word2ind']['UNK']) 
                                  for word in words], dtype='int64')
tokenize = lambda string: ['<START>'] + word_tokenize(string) + ['<END>']
to_str_pred = lambda w, l: str(" ".join([info['ind2word'][x] for x in list( filter(
    lambda x:x>0,w.data.cpu().numpy()))][:l.data.cpu()[0]]))[8:]
to_str_gt = lambda w: str(" ".join([info['ind2word'][x] for x in filter(
    lambda x:x>0,w.data.cpu().numpy())]))[8:-6]
def var_map(tensor):
    if params['useGPU']:
        tensor = tensor.cuda()
    return Variable(tensor.unsqueeze(0), volatile=True)

def fetch_q_bot_response(qBot, history, caption, **generation_kwargs):
    caption_tokens = ind_map(tokenize(caption))
    h_tokens = []
    for item in history:
        tokens = ind_map(tokenize(item['text']))
        h_tokens.append((item['speaker'], tokens,))
    caption_tensor = var_map(torch.from_numpy(caption_tokens))
    caption_lens = var_map(torch.LongTensor([len(caption_tokens)]))

    hist_tensors = [var_map(torch.from_numpy(x[1])) for x in h_tokens]
    hist_lens = [var_map(torch.LongTensor([len(x[1])])) for x in h_tokens]

    qBot.eval(), qBot.reset()
    qBot.observe(-1, caption=caption_tensor, captionLens=caption_lens)
    for i in range(len(h_tokens)):
        round = i // 2
        if h_tokens[i][0] == 'question':
            qBot.observe(round, ques=hist_tensors[i], quesLens=hist_lens[i])
        elif h_tokens[i][0] == 'answer':
            qBot.observe(round, ans=hist_tensors[i], ansLens=hist_lens[i])
        else:
            raise NotImplementedError
        qBot.encoder.embedInputDialog()
    questions, quesLens = qBot.forwardDecode(**generation_kwargs)
    pred_str = to_str_pred(questions[0], quesLens[0])
    return pred_str

def fetch_reward(qBot, history, img_features, caption):
    img_features = var_map(img_features)
    caption_tokens = ind_map(tokenize(caption))
    h_tokens = []
    for item in history:
        tokens = ind_map(tokenize(item['text']))
        h_tokens.append((item['speaker'], tokens,))
    caption_tensor = var_map(torch.from_numpy(caption_tokens))
    caption_lens = var_map(torch.LongTensor([len(caption_tokens)]))

    hist_tensors = [var_map(torch.from_numpy(x[1])) for x in h_tokens]
    hist_lens = [var_map(torch.LongTensor([len(x[1])])) for x in h_tokens]

    qBot.eval(), qBot.reset()
    qBot.observe(-1, caption=caption_tensor, captionLens=caption_lens)

    distances = [F.mse_loss(qBot.predictImage(), img_features)]
    for i in range(len(h_tokens)):
        round = i // 2
        if h_tokens[i][0] == 'question':
            qBot.observe(round, ques=hist_tensors[i], quesLens=hist_lens[i])
            qBot.encoder.embedInputDialog()
        elif h_tokens[i][0] == 'answer':
            qBot.observe(round, ans=hist_tensors[i], ansLens=hist_lens[i])
            distances.append(F.mse_loss(qBot.predictImage(), img_features))
        else:
            raise NotImplementedError
    rewards = list(map(lambda x: x[1]-x[0], zip(distances[:-1], distances[1:])))
    return rewards[-1]

def fetch_a_bot_response(aBot, history, img_features, caption, **generation_kwargs): 
    img_features = var_map(img_features)
    caption_tokens = ind_map(tokenize(caption))
    h_tokens = []
    for item in history:
        tokens = ind_map(tokenize(item['text']))
        h_tokens.append((item['speaker'], tokens,))
    caption_tensor = var_map(torch.from_numpy(caption_tokens))
    caption_lens = var_map(torch.LongTensor([len(caption_tokens)]))

    hist_tensors = [var_map(torch.from_numpy(x[1])) for x in h_tokens]
    hist_lens = [var_map(torch.LongTensor([len(x[1])])) for x in h_tokens]

    aBot.eval(), aBot.reset()
    aBot.observe(-1, image=img_features, caption=caption_tensor, captionLens=caption_lens)

    for i in range(len(h_tokens)):
        round = i // 2
        if h_tokens[i][0] == 'question':
            aBot.observe(round, ques=hist_tensors[i], quesLens=hist_lens[i])
        elif h_tokens[i][0] == 'answer':
            aBot.observe(round, ans=hist_tensors[i], ansLens=hist_lens[i])
        else:
            raise NotImplementedError
        aBot.encoder.embedInputDialog()
    answers, ansLens = aBot.forwardDecode(**generation_kwargs)
    pred_str = to_str_pred(answers[0], ansLens[0])
    return pred_str


if __name__ == "__main__":
    while True:
        item = random.choice(dataset)
        img_feat = item['img_feat']
        caption = to_str_gt(item['cap'])
        data = []
        n_rounds = 10
        print('='*25)
        print('caption:', caption)
        print()

        for _ in range(n_rounds):
            q_response = fetch_q_bot_response(qBot, data, caption, inference='greedy', beamSize=1)
            print('q:', q_response)
            data.append({'speaker': 'question', 'text': q_response})
            a_response = fetch_a_bot_response(aBot, data, img_feat, caption, inference='greedy', beamSize=1)
            print('a:', a_response)
            data.append({'speaker': 'answer', 'text': a_response})
            print('reward:', fetch_reward(qBot, data, img_feat, caption).item())
        print('='*25)
