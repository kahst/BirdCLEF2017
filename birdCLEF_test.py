#!/usr/bin/env python

print "HANDLING IMPORTS...",

import os
import time
import random
import operator
import argparse

import numpy as np
import cv2

from sklearn.utils import shuffle
import itertools

import scipy.io.wavfile as wave
from scipy import interpolate

import python_speech_features as psf
from pydub import AudioSegment

import pickle

import theano
import theano.tensor as T

from lasagne import random as lasagne_random
from lasagne import layers as l
from lasagne import nonlinearities
from lasagne import init
from lasagne import objectives
from lasagne import updates
from lasagne import regularization

try:
    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
except ImportError:
    from lasagne.layers import BatchNormLayer    

print "DONE!"

######################## CONFIG #########################
#Fixed random seed
RANDOM_SEED = 1337
RANDOM = np.random.RandomState(RANDOM_SEED)
lasagne_random.set_rng(RANDOM)

#Image params
IM_SIZE = (512, 256) #(width, height)
IM_DIM = 1

#General model params
MODEL_TYPE = 1
MULTI_LABEL = False
NONLINEARITY = nonlinearities.elu #nonlinearities.rectify
INIT_GAIN = 1.0 #1.0 if elu, sqrt(2) if rectify

#Pre-trained model params
MODEL_PATH = 'model/'
PRETRAINED_MODEL = 'birdCLEF_TUCMI_Run1_Model.pkl'

#We need to define the class labels our net has learned
#but we use another file for that
from birdCLEF_class_labels import CLASSES

################### ARGUMENT PARSER #####################
def parse_args():
    
    parser = argparse.ArgumentParser(description='BirdCLEF bird sound classification')
    parser.add_argument('--filename', dest='filename', help='path to sample wav file for testing', type=str, default='')
    parser.add_argument('--overlap', dest='spec_overlap', help='spectrogram overlap in seconds', type=int, default=0)
    parser.add_argument('--results', dest='num_results', help='number of results', type=int, default=5)
    parser.add_argument('--confidence', dest='min_confidence', help='confidence threshold', type=float, default=0.01)

    args = parser.parse_args()

    return args

################ SPECTROGRAM EXTRACTION #################
#Change sample rate if not 44.1 kHz
def changeSampleRate(sig, rate):

    duration = sig.shape[0] / rate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * 44100 / rate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    new_audio = interpolator(time_new).T

    sig = np.round(new_audio).astype(sig.dtype)
    
    return sig, 44100

#Get magnitude spec from signal split
def getMagSpec(sig, rate, winlen, winstep, NFFT):

    #get frames
    winfunc = lambda x:np.ones((x,))
    frames = psf.sigproc.framesig(sig, winlen*rate, winstep*rate, winfunc)        
    
    #Magnitude Spectrogram
    magspec = np.rot90(psf.sigproc.magspec(frames, NFFT))

    return magspec

#Split signal into five-second chunks with overlap of 4 and minimum length of 1 second
#Use these settings for other chunk lengths:
#winlen, winstep, seconds
#0.05, 0.0097, 5s
#0.05, 0.0195, 10s
#0.05, 0.0585, 30s
def getMultiSpec(path, seconds=5, overlap=2, minlen=1, winlen=0.05, winstep=0.0097, NFFT=840):

    #open wav file
    (rate,sig) = wave.read(path)

    #adjust to different sample rates
    if rate != 44100:
        sig, rate = changeSampleRate(sig, rate)

    #split signal with overlap
    sig_splits = []
    for i in xrange(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + seconds * rate]
        if len(split) >= minlen * rate:
            sig_splits.append(split)

    #is signal too short for segmentation?
    if len(sig_splits) == 0:
        sig_splits.append(sig)

    #calculate spectrogram for every split
    for sig in sig_splits:

        #preemphasis
        sig = psf.sigproc.preemphasis(sig, coeff=0.95)

        #get spec
        magspec = getMagSpec(sig, rate, winlen, winstep, NFFT)

        #get rid of high frequencies
        h, w = magspec.shape[:2]
        magspec = magspec[h - 256:, :]

        #normalize in [0, 1]
        magspec -= magspec.min(axis=None)
        magspec /= magspec.max(axis=None)        

        #fix shape to 512x256 pixels without distortion
        magspec = magspec[:256, :512]
        temp = np.zeros((256, 512), dtype="float32")
        temp[:magspec.shape[0], :magspec.shape[1]] = magspec
        magspec = temp.copy()
        magspec = cv2.resize(magspec, (512, 256))
        
        #DEBUG: show spec
        #cv2.imshow('SPEC', magspec)
        #cv2.waitKey(-1)

        yield magspec

################## BUILDING THE MODEL ###################
def buildModel(mtype=1):

    print "BUILDING MODEL TYPE", mtype, "..."

    #default settings (Model 1)
    filters = 64
    first_stride = 2
    last_filter_multiplier = 16

    #specific model type settings (see working notes for details)
    if mtype == 2:
        first_stride = 1
    elif mtype == 3:
        filters = 32
        last_filter_multiplier = 8

    #input layer
    net = l.InputLayer((None, IM_DIM, IM_SIZE[1], IM_SIZE[0]))

    #conv layers
    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters, filter_size=7, pad='same', stride=first_stride, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    if mtype == 2:
        net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters, filter_size=5, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
        net = l.MaxPool2DLayer(net, pool_size=2)

    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters * 2, filter_size=5, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters * 4, filter_size=3, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters * 8, filter_size=3, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    net = l.batch_norm(l.Conv2DLayer(net, num_filters=filters * last_filter_multiplier, filter_size=3, pad='same', stride=1, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.MaxPool2DLayer(net, pool_size=2)

    print "\tFINAL POOL OUT SHAPE:", l.get_output_shape(net) 

    #dense layers
    net = l.batch_norm(l.DenseLayer(net, 512, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))
    net = l.batch_norm(l.DenseLayer(net, 512, W=init.HeNormal(gain=INIT_GAIN), nonlinearity=NONLINEARITY))

    #Classification Layer
    if MULTI_LABEL:
        net = l.DenseLayer(net, NUM_CLASSES, nonlinearity=nonlinearities.sigmoid, W=init.HeNormal(gain=1))
    else:
        net = l.DenseLayer(net, NUM_CLASSES, nonlinearity=nonlinearities.softmax, W=init.HeNormal(gain=1))

    print "...DONE!"

    #model stats
    print "MODEL HAS", (sum(hasattr(layer, 'W') for layer in l.get_all_layers(net))), "WEIGHTED LAYERS"
    print "MODEL HAS", l.count_params(net), "PARAMS"

    return net

NUM_CLASSES = len(CLASSES)
NET = buildModel(MODEL_TYPE)

####################  MODEL LOAD  ########################
def loadParams(epoch, filename=None):
    print "IMPORTING MODEL PARAMS...",
    net_filename = MODEL_PATH + filename
    with open(net_filename, 'rb') as f:
        params = pickle.load(f)
    l.set_all_param_values(NET, params)
    print "DONE!"

#load params of trained model
loadParams(-1, filename=PRETRAINED_MODEL)

################# PREDICTION FUNCTION ####################
def getPredictionFuntion(net):
    net_output = l.get_output(net, deterministic=True)

    print "COMPILING THEANO TEST FUNCTION...",
    start = time.time()
    test_net = theano.function([l.get_all_layers(NET)[0].input_var], net_output, allow_input_downcast=True)
    print "DONE! (", int(time.time() - start), "s )"

    return test_net

TEST_NET = getPredictionFuntion(NET)

################# PREDICTION POOLING ####################
def predictionPooling(p):

    #You can test different prediction pooling strategies here
    #We only use average pooling
    p_pool = np.mean(p, axis=0)

    return p_pool

####################### PREDICT #########################
def predict(img):    

    #transpose image if dim=3
    try:
        img = np.transpose(img, (2, 0, 1))
    except:
        pass

    #reshape image
    img = img.reshape(-1, IM_DIM, IM_SIZE[1], IM_SIZE[0])

    #calling the test function returns the net output
    prediction = TEST_NET(img)[0] 

    return prediction

####################### TESTING #########################
def testFile(path, spec_overlap=4, num_results=5, confidence_threshold=0.01):

    #time
    start = time.time()
    
    #extract spectrograms from wav-file and process them
    predictions = []
    spec_cnt = 0
    for spec in getMultiSpec(path, overlap=spec_overlap, minlen=1):

        #make prediction
        p = predict(spec)
        spec_cnt += 1

        #stack predictions
        if len(predictions):
            predictions = np.vstack([predictions, p])  
        else:
            predictions = p

    #prediction pooling
    p_pool = predictionPooling(predictions)

    #get class labels for predictions
    p_labels = {}
    for i in range(p_pool.shape[0]):
        if p_pool[i] >= confidence_threshold:
            p_labels[CLASSES[i]] = p_pool[i]

    #sort by confidence and limit results (None returns all results)
    p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)[:num_results]

    #take time again
    dur = time.time() - start

    return p_sorted, spec_cnt, dur

#################### EXAMPLE USAGE ######################
if __name__ == "__main__":

    #adjust config
    args = parse_args()

    #do testing
    print 'TESTING:', args.filename
    pred, cnt, dur = testFile(args.filename, args.spec_overlap, args.num_results, args.min_confidence)    
    print 'TOP PREDICTION(S):'
    for p in pred:
        print '\t', p[0], int(p[1] * 100), '%'
    print 'PREDICTION FOR', cnt, 'SPECS TOOK', int(dur * 1000), 'ms (', int(dur / cnt * 1000) , 'ms/spec', ')'




