#!/usr/bin/env python
# -*- coding: utf-8 -*-

print "HANDLING IMPORTS..."

import warnings
warnings.filterwarnings('ignore')

import os
import time
import json
import operator

import traceback
import numpy as np
import cv2

import xmltodict as x2d

import pickle

from sklearn.utils import shuffle

import scipy.ndimage as ndimage
import scipy.stats as stats
from scipy import interpolate

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

import python_speech_features as psf
import scipy.io.wavfile as wave

print "...DONE!"

######################## CONFIG #########################
#Dataset params
XML_DIR = 'birdCLEF2017/xml/'
TRAIN_DIR = 'dataset/train/spec_44.1/'
TEST_DIR = 'dataset/val/src/'

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
TRAINED_MODEL = 'birdCLEF_TUCMI_Run1_Model.pkl'
LOAD_OUTPUT_LAYER = True

#Testing params
CLASS_RANGE = [None, None]
MAX_SAMPLES = None
BATCH_SIZE = 128 #Choose BATCH_SIZE = 1 for soundscapes -> results will have timestamps
SPEC_OVERLAP = 4 #Choose OVERLAP = 0 for soundscapes
SPEC_LENGTH = 5
PREEMPHASIS = 0.95
CONFIDENCE_THRESHOLD = 0.0001
INCLUDE_BG_SPECIES = True
ONLY_BG_SPECIES_FROM_CLASSES = True
MAX_PREDICTIONS = 100
RESULT_FILE = 'birdCLEF_eval_result.txt'

#Prediction save/load for ensemble testing
EVAL_ID = 'example_eval_id'
SAVE_PREDICTIONS = True
LOAD_EVAL_ID_PREDICTIONS = None #'previous_eval_id'

#List of classes the mdoel ist trained on; use print classes in birdCLEF_train.py for listing; [] = all classes from dataset/train
CLS = []

################### AUDIO PROCESSING ####################
def getGroundTruth(classes):

    gt = {}
    bg_species = []

    #status
    print "PARSING XML FILES...",

    #prepare class labels
    labels = []
    for c in classes:
        labels.append(c.rsplit(" ", 1)[0])

    #open every xml-file
    for xmlp in sorted(os.listdir(XML_DIR)):

        xml = open(XML_DIR + xmlp, 'r').read()
        xmldata = x2d.parse(xml)

        #get primary species
        primary =  xmldata['Audio']['Genus'] + ' ' + xmldata['Audio']['Species']

        #get background species
        if 'BackgroundSpecies' in xmldata['Audio'] and xmldata['Audio']['BackgroundSpecies']:
            background = xmldata['Audio']['BackgroundSpecies'].split(",")
        else:
            background = []

        #only bg species from classes?
        bg = []
        for bgs in background:
            if bgs in labels or not ONLY_BG_SPECIES_FROM_CLASSES:
                bg.append(bgs)
        background = bg

        #background species stats
        for bgs in background:
            if bgs not in bg_species:
                bg_species.append(bgs)

        #compose gt entry
        gt[xmlp.rsplit(".", 1)[0]] = {'primary': primary, 'background':background}

    #status
    print "DONE! (FOUND", len(bg_species), "BACKGROUND SPECIES)"

    return gt

def parseTestSet():    

    #get classes of trainig set (subfolders as class lables; has to be same as during training, shuffled or alphabetically)
    classes = [folder for folder in sorted(os.listdir(TRAIN_DIR))][CLASS_RANGE[0]:CLASS_RANGE[1]]
    cls_index = classes

    #Only use specific classes?
    if len(CLS) > 0:
        classes = CLS

    #load ground truth
    gt = getGroundTruth(classes)

    #get list of test files
    test = []
    test_classes = [os.path.join(TEST_DIR, tc) for tc in sorted(os.listdir(TEST_DIR))]
    for tc in test_classes:
        if tc.rsplit("/", 1)[-1] in classes:
            test += [os.path.join(tc, fpath) for fpath in os.listdir(tc)]
    test = shuffle(test, random_state=RANDOM)[:MAX_SAMPLES]

    #stats
    #print classes
    print "NUMBER OF CLASSES:", len(classes)
    print "NUMBER OF TEST SAMPLES:", len(test)

    return gt, test, classes, cls_index    
    
GT, TEST, CLASSES, CLASS_INDEX = parseTestSet()
NUM_CLASSES = len(CLASSES)
################### AUDIO PROCESSING ####################
def changeSampleRate(sig, rate):

    duration = sig.shape[0] / rate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * 44100 / rate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    new_audio = interpolator(time_new).T

    sig = np.round(new_audio).astype(sig.dtype)
    
    return sig, 44100

def getSpecFromSignal(sig, rate, winlen=0.05, winstep=0.0097, NFFT=840):

    if SPEC_LENGTH == 10:
        winstep = 0.0195
    elif SPEC_LENGTH == 30:
        winstep = 0.0585

    #preemphasis
    sig = psf.sigproc.preemphasis(sig, coeff=PREEMPHASIS)

    #get frames
    winfunc=lambda x:np.ones((x,))
    frames = psf.sigproc.framesig(sig, winlen*rate, winstep*rate, winfunc)        
    
    #Magnitude Spectrogram
    magspec = np.rot90(psf.sigproc.magspec(frames, NFFT))

    #crop frequency
    h, w = magspec.shape[:2]
    magspec = magspec[h - 256:, :]

    #normalization
    magspec -= magspec.min(axis=None)
    magspec /= magspec.max(axis=None)    

    #adjust shape
    magspec = magspec[:256, :512]
    temp = np.zeros((256, 512), dtype="float32")
    temp[:magspec.shape[0], :magspec.shape[1]] = magspec
    magspec = temp.copy()
    magspec = cv2.resize(magspec, (IM_SIZE[0], IM_SIZE[1]))
    
    #show
    #cv2.imshow('SPEC', magspec)
    #cv2.waitKey(-1)

    #reshape
    magspec = magspec.reshape(-1, IM_DIM, IM_SIZE[1], IM_SIZE[0])

    return magspec

#################### BATCH HANDLING #####################
def getSignalChunk(sig, rate, seconds=SPEC_LENGTH):

    #split signal with overlap
    sig_splits = []
    for i in xrange(0, len(sig), int((seconds - SPEC_OVERLAP) * rate)):
        split = sig[i:i + seconds * rate]
        if len(split) >= seconds / 2 * rate:
            sig_splits.append([split, i / rate, i / rate + seconds])

    #is signal too short for segmentation?
    if len(sig_splits) == 0:
        sig_splits.append([sig, 0, seconds])

    #get batch-sized chunks of image paths
    for i in xrange(0, len(sig_splits), BATCH_SIZE):
        yield sig_splits[i:i+BATCH_SIZE]

def getNextSpecBatch(path):

    #open wav file
    (rate, sig) = wave.read(path)

    #change sample rate if needed
    if rate != 44100:
        sig, rate = changeSampleRate(sig, rate)

    #fill batches
    for sig_chunk in getSignalChunk(sig, rate):

        #allocate numpy arrays for image data and targets
        s_b = np.zeros((BATCH_SIZE, IM_DIM, IM_SIZE[1], IM_SIZE[0]), dtype='float32')
        s_t = np.zeros((BATCH_SIZE, 2), dtype='float32')

        ib = 0
        for s in sig_chunk:
        
            #load spectrogram data from sig
            spec = getSpecFromSignal(s[0], rate)

            #pack into batch array
            s_b[ib] = spec
            s_t[ib][0] = s[1]
            s_t[ib][1] = s[2]
            ib += 1

        #trim to actual size
        s_b = s_b[:ib]
        s_t = s_t[:ib]

        #yield batch
        yield s_b, s_t

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

NET = buildModel(MODEL_TYPE)
####################  MODEL LOAD  ########################
def loadParams(epoch, filename=None):
    print "IMPORTING MODEL PARAMS...",
    net_filename = MODEL_PATH + filename
    with open(net_filename, 'rb') as f:
        params = pickle.load(f)
    if LOAD_OUTPUT_LAYER:
        l.set_all_param_values(NET, params)
    else:
        l.set_all_param_values(l.get_all_layers(NET)[:-1], params[:-2])
    print "DONE!"

################  PREDICTION SAVE/LOAD  ##################
def savePredictions(pred):    
    print "EXPORTING PREDICTIONS...",
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    p_filename = 'predictions/' + EVAL_ID + '_predictions.pkl'
    with open(p_filename, 'w') as f:
        pickle.dump(pred, f) 
    print "DONE!"

def loadPredictions(eval_id):
    print "IMPORTING PREDICTIONS...",
    p_filename = 'predictions/' + eval_id + '_predictions.pkl'
    with open(p_filename, 'rb') as f:
        pred = pickle.load(f)
    print "DONE!"

    return pred

################# PREDICTION FUNCTION ####################
def getPredictionFuntion(net):
    net_output = l.get_output(net, deterministic=True)

    print "COMPILING THEANO TEST FUNCTION...",
    start = time.time()
    test_net = theano.function([l.get_all_layers(net)[0].input_var], net_output, allow_input_downcast=True)
    print "DONE! (", int(time.time() - start), "s )"

    return test_net

################# PREDICTION POOLING ####################
def predictionPooling(p):

    #You can test different prediction pooling strategies here
    #We only use average pooling
    p_pool = np.mean(p, axis=0)

    return p_pool

def calcAvgPrecision(p, gt):

    precision = 0
    rcnt = 0

    #parse ordered predictions
    for i in range(0, len(p)):

        #relevant species
        if p[i][0] in gt:            
            rcnt += 1
            precision += float(rcnt) / (i + 1)

    #precision by relevant species from ground truth
    avgp = precision / len(gt)

    return avgp

def getTimeString(t):

    h = t // 3600
    t -= h * 3600
    m = t // 60
    t -= m * 60
    s = str(h).zfill(2) + ":" + str(m).zfill(2) + ":" + str(t).zfill(2)

    return s

def getTimecode(time_batch):

    #only if batch size == 1
    if BATCH_SIZE == 1:
        start, end = time_batch[0]
        return getTimeString(int(start)) + "-" + getTimeString(int(end))

    return ""

def makePrediction(path):

    #make predictions for batches of spectrograms
    prediction = []
    for spec_batch, time_batch in getNextSpecBatch(path):

        #predict
        p = test_net(spec_batch)

        #stack predictions
        if BATCH_SIZE > 1:
            if len(prediction):
                prediction = np.vstack([prediction, p])  
            else:
                prediction = p
        else:
            yield p, time_batch

    if BATCH_SIZE > 1:
        yield prediction, time_batch

####################### TESTING #########################
if LOAD_EVAL_ID_PREDICTIONS:
    ensemble_pred = loadPredictions(LOAD_EVAL_ID_PREDICTIONS)
else:
    ensemble_pred = {}

#test model
print "TESTING MODEL..."

#load model params
loadParams(-1, filename=TRAINED_MODEL)

#get test function
test_net = getPredictionFuntion(NET)

#open result file
rfile = open(RESULT_FILE, 'w')

pr = []
pcnt = 1

#test every sample from test collection
for path in TEST:

    #media id
    mid = int(path.split("/")[-1].split(".")[0].split("_RN")[1])

    #status
    print pcnt, path.replace(TEST_DIR, ''),

    try:

        #make predictions for batches of spectrograms
        for prediction, time_batch in makePrediction(path):
            
            #prediction pooling
            p_pool = predictionPooling(prediction)

            #empty prediction
            p_new = np.zeros((len(CLASS_INDEX)), dtype='float32')

            #fit pooled prediction into empty prediction based on CLASS_INDEX
            for i in range(0, p_pool.shape[0]):

                #class index
                index = CLASS_INDEX.index(CLASSES[i])

                #fit
                p_new[index] = p_pool[i]

            #save predictions
            fname = path.split("/")[-1] + getTimecode(time_batch)
            if not fname in ensemble_pred:
                ensemble_pred[fname] = []
            ensemble_pred[fname].append(p_new)

            #average prediction (only use non-zero elements)
            p_avg = np.divide(np.array(ensemble_pred[fname]).sum(axis=0), (np.array(ensemble_pred[fname]) > 0).sum(axis=0))
            
            #get top predictions
            p_top = {}
            p_id = {}
            for i in range(0, p_avg.shape[0]):
                if p_avg[i] > CONFIDENCE_THRESHOLD:
                    p_top[CLASS_INDEX[i].rsplit(' ', 1)[0]] = p_avg[i]
                    p_id[CLASS_INDEX[i].rsplit(' ', 1)[1]] = p_avg[i] 
            
            #order top predictions
            p_top_sorted = sorted(p_top.items(), key=operator.itemgetter(1), reverse=True)[:MAX_PREDICTIONS]
            p_id_sorted = sorted(p_id.items(), key=operator.itemgetter(1), reverse=True)[:MAX_PREDICTIONS]

            #get ground truth annotations
            fileid = path.split('/')[-1].split('_', 1)[1].rsplit('.', 1)[0]
            species = [GT[fileid]['primary']]
            if INCLUDE_BG_SPECIES:
                species += GT[fileid]['background']

            #calculate AVGPrecision
            avgp = calcAvgPrecision(p_top_sorted, species)
            pr.append(avgp)

            #print result
            print "AVGP", avgp, "MAP:", np.mean(pr)

            #task result compilation
            for s in p_id_sorted:
                rfile.write(str(mid) + ";" + getTimecode(time_batch) + ";" + s[0] + ";" + str(s[1]) + "\n")
    
    except KeyboardInterrupt:
        break
    except:
        print "ERROR"
        pr.append(0.0)
        traceback.print_exc()
        continue

    pcnt += 1

if SAVE_PREDICTIONS:
    savePredictions(ensemble_pred)
    
print "TESTING DONE!"
print "mAP:", np.mean(pr)

#close result file
rfile.close()

        

    
