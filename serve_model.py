from functools import partial
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

import redis
from flask import Flask, request
from flask_cors import CORS

import pickle as pkl
import time
import multiprocessing as mp
import requests
import re

Q = None

app = Flask(__name__)
CORS(app)

r = redis.Redis(host='localhost', port=6379, db=1)

# seed rng for reproducibility
# manualSeed = 1234
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)
# if params['useGPU']:
#     torch.cuda.manual_seed_all(manualSeed)

params, info, dataset, qBot, aBot = None, None, None, None, None

def load_objects():
    # read the command line options
    params = options.readCommandLine()

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
    return dataset, qBot, aBot, params, info


ind_map = lambda words: np.array([info['word2ind'].get(word, info['word2ind']['UNK']) 
                                for word in words], dtype='int64')
tokenize = lambda string: ['<START>'] + word_tokenize(string) + ['<END>']
to_str_pred = lambda w, l: str(" ".join([info['ind2word'][x] for x in list( filter(
    lambda x:x>0,w.data.cpu().numpy()))][:l.data.cpu()[0]]))[8:]
to_str_gt = lambda w: str(" ".join([info['ind2word'][x] for x in filter(
    lambda x:x>0,w.data.cpu().numpy())]))[8:-6]

def fix_tokenization_spaces(text):
    text = text.replace("' ", "'").replace(" '", "'")
    text = text.replace("ca n't", "can't").replace(" ?", "?")
    text = text.replace("do n't", "don't").replace(" ,", ",")
    text = text.replace("is n't", "isn't")
    return text

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

def step(qBot, aBot, history, img_features, caption, **generation_kwargs):
    a_response = fetch_a_bot_response(aBot, history, img_features, caption, **generation_kwargs)
    history += [{'speaker': 'answer', 'text': a_response}]
    reward = fetch_reward(qBot, history, img_features, caption)
    return a_response, reward

def model_process():
    global qBot
    global aBot
    global params
    global info
    global dataset
    dataset, qBot, aBot, params, info = load_objects()
    print('BOTS LOADED!')
    while True:
        request_id, f_name, args, kwargs = Q.get()
        if f_name == 'fetch_a_bot_response':
            f = partial(fetch_a_bot_response, aBot)
        elif f_name == 'fetch_q_bot_response':
            f = partial(fetch_q_bot_response, qBot)
        elif f_name == 'fetch_reward':
            f = partial(fetch_reward, qBot)
        elif f_name == 'step':
            f = partial(step, qBot, aBot)
        else:
            raise NotImplementedError
        result = f(*args, **kwargs)
        r.set('result_%d' % (request_id), pkl.dumps(result))

def flask_process():
    app.run(host='0.0.0.0', port=5000, threaded=True, processes=1)

@app.route('/fetch_a_bot_response', methods=['POST'])
def flask_fetch_a_bot_response():
    history = request.form.get('history', None)
    img_features = request.form.get('img_features', None)
    caption = request.form.get('caption', None)
    generation_kwargs = request.form.get('generation_kwargs', None)
    history = json.loads(history)
    img_features = torch.tensor(json.loads(img_features))
    generation_kwargs = json.loads(generation_kwargs)

    request_id = int(r.incr('request_id_counter'))
    Q.put((request_id, 'fetch_a_bot_response', (history, img_features, caption,), generation_kwargs,))
    while not r.exists("result_%d" % (request_id)):
        time.sleep(0.05)
    
    result = pkl.loads(r.get("result_%d" % (request_id)))
    r.delete("result_%d" % (request_id))
    return json.dumps(fix_tokenization_spaces(result))

@app.route('/fetch_q_bot_response', methods=['POST'])
def flask_fetch_q_bot_response():
    history = request.form.get('history', None)
    caption = request.form.get('caption', None)
    generation_kwargs = request.form.get('generation_kwargs', None)
    history = json.loads(history)
    generation_kwargs = json.loads(generation_kwargs)

    request_id = int(r.incr('request_id_counter'))
    Q.put((request_id, 'fetch_q_bot_response', (history, caption,), generation_kwargs,))
    while not r.exists("result_%d" % (request_id)):
        time.sleep(0.05)
    
    result = pkl.loads(r.get("result_%d" % (request_id)))
    r.delete("result_%d" % (request_id))
    return json.dumps(fix_tokenization_spaces(result))

@app.route('/fetch_reward', methods=['POST'])
def flask_fetch_reward():
    history = request.form.get('history', None)
    img_features = request.form.get('img_features', None)
    caption = request.form.get('caption', None)
    history = json.loads(history)
    img_features = torch.tensor(json.loads(img_features))

    request_id = int(r.incr('request_id_counter'))
    Q.put((request_id, 'fetch_reward', (history, img_features, caption,), {},))
    while not r.exists("result_%d" % (request_id)):
        time.sleep(0.05)
    
    result = pkl.loads(r.get("result_%d" % (request_id)))
    r.delete("result_%d" % (request_id))
    return json.dumps(result.item())

@app.route('/step', methods=['POST'])
def flask_step():
    history = request.form.get('history', None)
    img_features = request.form.get('img_features', None)
    caption = request.form.get('caption', None)
    generation_kwargs = request.form.get('generation_kwargs', None)
    history = json.loads(history)
    img_features = torch.tensor(json.loads(img_features))
    generation_kwargs = json.loads(generation_kwargs)

    request_id = int(r.incr('step'))
    Q.put((request_id, 'step', (history, img_features, caption,), generation_kwargs,))
    while not r.exists("result_%d" % (request_id)):
        time.sleep(0.05)
    
    result = pkl.loads(r.get("result_%d" % (request_id)))
    r.delete("result_%d" % (request_id))
    return json.dumps((fix_tokenization_spaces(result[0]), result[1].item(),))

def main():
    global Q
    Q = mp.Manager().Queue()

    p = mp.Process(target=flask_process)
    p.start()

    model_process()

def test():
    global params
    global info
    global dataset
    dataset, _, _, params, info = load_objects()
    url = 'http://172.31.10.99:5000/'
    while True:
        item = random.choice(dataset)
        img_feat = item['img_feat'].tolist()
        caption = to_str_gt(item['cap'])
        data = []
        n_rounds = 10
        print('='*25)
        print('caption:', caption)
        print()

        for _ in range(n_rounds):
            q_response = json.loads(requests.post(url+'fetch_q_bot_response', 
                                                  data={'history': json.dumps(data), 
                                                        'caption': caption, 
                                                        'generation_kwargs': json.dumps({'inference': 'greedy', 'beamSize': 1})}).text)
            print('q:', q_response)
            data.append({'speaker': 'question', 'text': q_response})
            a_response, reward = json.loads(requests.post(url+'step', 
                                                          data={'history': json.dumps(data), 
                                                                'caption': caption, 
                                                                'img_features': json.dumps(img_feat), 
                                                                'generation_kwargs': json.dumps({'inference': 'greedy', 'beamSize': 1})}).text)
            print('a:', a_response)
            data.append({'speaker': 'answer', 'text': a_response})
            print('reward:', reward)
        print('='*25)

if __name__ == "__main__":
    main()
    # test()
