'''
This code is inspired from Plug and Play generative models paper (PPGN) Link : https://github.com/Evolving-AI-Lab/ppgn
'''

import os, sys
os.environ['GLOG_minloglevel'] = '2'    # suprress Caffe verbose prints

import settings
sys.path.insert(0, settings.caffe_root)
import caffe

import numpy as np
from numpy.linalg import norm
import scipy.misc, scipy.io
import argparse 
import util
from sampler_multi import Sampler
import re
import random

if settings.gpu:
    caffe.set_mode_gpu() # sampling on GPU (recommended for speed) 

class CaptionConditionalSampler(Sampler):

    def __init__ (self, lstm_definition, lstm_weights):

        self.lstm = caffe.Net(lstm_definition, lstm_weights, caffe.TEST)

    def forward_backward_from_x_to_condition(self, net, end, image, condition):
        '''
        Forward and backward passes through 'net', the condition model p(y|x), here an image classifier. 
        '''

        src = net.blobs['data'] # input image
        dst = net.blobs[end]

        sentence = condition
        
        previous_word = 0

        lstm_layer = "log_prob"
        feature_layer = "image_features"

        grad_sum = np.zeros_like(self.lstm.blobs[feature_layer].data) 
        probs = []


        for idx, word in enumerate(sentence):
            if idx > 0:
                previous_word = sentence[idx - 1]

            # preparing lstm feature vectors
            cont = 0 if previous_word == 0 else 1
            cont_input = np.array([cont])
            word_input = np.array([previous_word])  # Previous word == 0 : meaning this is the start of the sentence

            # 1. Get feature descriptors from fc8
            net.forward(data=image, end=end)
            descriptor = net.blobs[end].data

            # 2. Pass this to lstm
            image_features = np.zeros_like(self.lstm.blobs[feature_layer].data)
            image_features[:] = descriptor

            self.lstm.forward(image_features=image_features, cont_sentence=cont_input,
                                        input_sentence=word_input, end=lstm_layer)

            # Display the prediction
            probs.append ( self.lstm.blobs["probs"].data[0,idx, word] )
            self.lstm.blobs[lstm_layer].diff[:, :, word] = 1

            diffs = self.lstm.backward(start=lstm_layer, diffs=[feature_layer])
            g_word = diffs[feature_layer]    # (1000,)

            grad_sum += g_word  # accumulate the gradient from all words 

            # reset objective after each step
            self.lstm.blobs[lstm_layer].diff.fill(0.)

        # Average softmax probabilities of all words
        obj_prob = np.mean(probs)

        # Backpropagate the gradient from LSTM to the feature extractor convnet
        dst.diff[...] = grad_sum[0]
        net.backward(start=end)
        g = src.diff.copy()

        dst.diff.fill(0.)   # reset objective after each step

        # Info to be printed out in the below 'print_progress' method
        info = { }
        return g, obj_prob, info 


    def get_label(self, condition):
        return None


    def print_progress(self, i, info, prob, grad):
        print "step: %04d\t [%.2f]\t norm: [%.2f]" % ( i, prob, norm(grad) )


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sentence', metavar='w', type=str, default="", nargs='?', help='Sentence to condition on')
    parser.add_argument('--n_iters', metavar='iter', type=int, default=10, help='Number of sampling steps per each unit')
    parser.add_argument('--threshold', metavar='w', type=float, default=-1.0, nargs='?', help='The probability threshold to decide whether to keep an image')
    parser.add_argument('--save_every', metavar='save_iter', type=int, default=1, help='Save a sample every N iterations. 0 to disable saving')
    parser.add_argument('--reset_every', metavar='reset_iter', type=int, default=0, help='Reset the code every N iterations')
    parser.add_argument('--lr', metavar='lr', type=float, default=2.0, nargs='?', help='Learning rate')
    parser.add_argument('--lr_end', metavar='lr', type=float, default=-1.0, nargs='?', help='Ending Learning rate')
    parser.add_argument('--epsilon2', metavar='lr', type=float, default=1.0, nargs='?', help='Ending Learning rate')
    parser.add_argument('--epsilon1', metavar='lr', type=float, default=1.0, nargs='?', help='Ending Learning rate')
    parser.add_argument('--epsilon3', metavar='lr', type=float, default=1.0, nargs='?', help='Ending Learning rate')
    parser.add_argument('--seed', metavar='n', type=int, default=0, nargs='?', help='Random seed')
    parser.add_argument('--xy', metavar='n', type=int, default=0, nargs='?', help='Spatial position for conv units')
    parser.add_argument('--opt_layer', metavar='s', type=str, help='Layer at which we optimize a code')
    parser.add_argument('--act_layer', metavar='s', type=str, default="fc8", help='Layer at which we activate a neuron')
    parser.add_argument('--init_file', metavar='s', type=str, default="None", help='Init image')
    parser.add_argument('--write_labels', action='store_true', default=False, help='Write class labels to images')
    parser.add_argument('--output_dir', metavar='b', type=str, default=".", help='Output directory for saving results')
    parser.add_argument('--net_weights', metavar='b', type=str, default=settings.encoder_weights, help='Weights of the net being visualized')
    parser.add_argument('--net_definition', metavar='b', type=str, default=settings.encoder_definition, help='Definition of the net being visualized')
    parser.add_argument('--captioner_definition', metavar='b', type=str, help='Definition of the net being visualized')
    parser.add_argument('--input_caption_file', metavar='b', type=str, help='Text file with multiple captions')

    args = parser.parse_args()

    # Default to constant learning rate
    if args.lr_end < 0:
        args.lr_end = args.lr

    # summary
    print "-------------"
    print " sentence: %s" % args.sentence
    print " n_iters: %s" % args.n_iters
    print " reset_every: %s" % args.reset_every
    print " save_every: %s" % args.save_every
    print " threshold: %s" % args.threshold

    print " epsilon1: %s" % args.epsilon1
    print " epsilon2: %s" % args.epsilon2
    print " epsilon3: %s" % args.epsilon3

    print " start learning rate: %s" % args.lr
    print " end learning rate: %s" % args.lr_end
    print " seed: %s" % args.seed
    print " opt_layer: %s" % args.opt_layer
    print " act_layer: %s" % args.act_layer
    print " init_file: %s" % args.init_file
    print "-------------"
    print " output dir: %s" % args.output_dir
    print " net weights: %s" % args.net_weights
    print " net definition: %s" % args.net_definition
    print " captioner definition: %s" % args.captioner_definition
    print "-------------"

    # encoder and generator for images 
    encoder = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)
    generator = caffe.Net(settings.generator_definition, settings.generator_weights, caffe.TEST)

    # condition network, here an image classification net
    # this LRCN image captioning net has 1 binary weights but 2 definitions: 1 for feature extractor (AlexNet), 1 for LSTM
    net = caffe.Net(args.net_definition, args.net_weights, caffe.TEST)
    
    # shape of the code being optimized
    shape = generator.blobs[settings.generator_in_layer].data.shape
    start_code = np.random.normal(0, 1, shape)
    
    file = open(args.input_caption_file,'r').readlines()
    
    conditions=[]
    for item in file: 
        strip_item = item.strip().lower()
        list_chars = ['\,','\.']
        for char in list_chars:
            pattern = re.compile(char)
            strip_item = re.sub(pattern,'',strip_item)
        
        if len(strip_item.split(' ')) >19: continue
        sentence = util.convert_words_into_numbers(settings.vocab_file, strip_item.split(' '))
        conditions.append({ "sentence": sentence, "readable":  item.strip()} )
 
    # Optimize a code via gradient ascent
    sampler = CaptionConditionalSampler(args.captioner_definition, args.net_weights)
    output_image, list_samples = sampler.sampling( condition_net=net, image_encoder=encoder, image_generator=generator, 
                        gen_in_layer=settings.generator_in_layer, gen_out_layer=settings.generator_out_layer, start_code=start_code, 
                        n_iters=args.n_iters, lr=args.lr, lr_end=args.lr_end, threshold=args.threshold, 
                        layer=args.act_layer, conditions=conditions,
                        epsilon1=args.epsilon1, epsilon2=args.epsilon2, epsilon3=args.epsilon3,
                        output_dir=args.output_dir, 
                        reset_every=args.reset_every, save_every=args.save_every)


    # Save the final image
    util.save_image(output_image, 'multi_caption.jpg')


if __name__ == '__main__':
    main()