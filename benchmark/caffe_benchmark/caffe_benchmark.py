# -*- coding: utf-8 -*-
"""
@author: Evgenii Vasiliev

Caffe classification benchmarking script 


Sample string to run benchmark: 

cd IXPUG_DNN_benchmark/caffe_benchmark
mkdir results_classification
mkdir results_detection
python3 caffe_benchmark.py -t classification -i ../datasets/imagenet/ -p ../models/resnet-50.prototxt -m ../models/resnet-50.caffemodel -ni 1000 -o False -of ./results_classification/ -r ./results_classification/result.csv
python3 caffe_benchmark.py -t detection -i ../datasets/pascal_voc/ -p ../models/ssd300.prototxt -m ../models/ssd300.caffemodel -ni 1000 -o False -of ./results_detection/ -r ./results_detection/result.csv -me [104,117,123]

Last modified 24.07.2019

"""

import cv2
import caffe
import os.path
import argparse
import numpy as np
from time import time

def build_argparser():
    parser=argparse.ArgumentParser()
    parser.add_argument('-p', '--proto', help='Path to an .prototxt \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-m', '--model', help='Path to an .caffemodel file \
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input_folder', help='Name of input folder',
        default='', type=str)
    parser.add_argument('-ni', '--number_iter', help='Number of inference \
        iterations', required=True, type=int)
    parser.add_argument('-o', '--output', help='Get output',
        required=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-of', '--output_folder', help='Name of output folder',
        default='', type=str)
    parser.add_argument('-r', '--result_file', help='Name of output folder', 
        default='result.csv', type=str)
    parser.add_argument('-t', '--task_type', help='Task type: \
        classification / detection', default = 'classification', type=str)
    parser.add_argument('-me', '--mean', help='Input mean values', 
                        default = '[0 0 0]', type=str)
    parser.add_argument('-b', '--batch_size', help='batch size', 
        required=True, type=int)
    return parser

def load_network(proto, model, batch_size = 1):
    caffe.set_mode_cpu()
    network = caffe.Net(proto, model, caffe.TEST)
    transformer = caffe.io.Transformer({'data': network.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    
    if batch_size > 1:
        channels = network.blobs['data'].data.shape[1]
        height = network.blobs['data'].data.shape[2]
        width = network.blobs['data'].data.shape[3]
        network.blobs['data'].reshape(batch_size, channels, height, width)
        network.reshape()
        
    return network, transformer

def load_images_to_network(image_paths, net, transformer):
    for i in range(len(image_paths)):
        im = caffe.io.load_image(image_paths[i])
        net.blobs['data'].data[i,:,:,:] = transformer.preprocess('data', im)
        
def load_images(transformer, input_folder, numbers):
    data = os.listdir(input_folder)
    counts = numbers
    if len(data)<numbers:
        counts = len(data)
    images = []
    for i in range(counts):
        im = caffe.io.load_image(os.path.join(input_folder, data[i]))
        images.append(transformer.preprocess('data', im))
    return images, counts

def prepare_image(image_path, input_dims, scale = 1.0, mean = [0.0, 0.0, 0.0]):
    image = cv2.imread(image_path, 1).astype(np.float32) - np.asarray(mean)
    image = image * scale
    image_size = image.shape
    image = cv2.resize(image, (input_dims[2], input_dims[3]))
    return image, image_size

def caffe_benchmark(net, transformer, number_iter, input_folder,
                    need_output = False, output_folder = '', task_type = '',
                    batch_size = 1):
    
    filenames = os.listdir(input_folder)
    inference_time = []
    
    number_iter = (number_iter + batch_size -1) // batch_size
    images, counts = load_images(transformer, input_folder, number_iter * batch_size)
    
    t0_total = time()
    for i in range(number_iter):
        
        for j in range(batch_size):
            net.blobs['data'].data[j,:,:,:] = images[i * batch_size+j]
        
        t0 = time()
        out = net.forward()
        t1 = time()
        
        if (need_output):
            if batch_size == 1:
                # Generate output name
                output_filename = str(os.path.splitext(os.path.basename(filenames[i]))[0])+'.npy'
                output_filename = os.path.join(os.path.dirname(output_folder), output_filename) 
                # Save output
                if task_type == 'classification':
                    classification_output(out, output_filename)
                elif task_type == 'detection':
                    detection_output(out, output_filename)
        inference_time.append(t1 - t0)
    t1_total = time()
    inference_total_time = t1_total - t0_total
    return out, inference_time, inference_total_time

def classification_output(prob, output_file):
    prob = prob['prob']
    prob = prob[0]
    np.savetxt(output_file, prob)

def detection_output(prob, output_file):
    prob = prob['detection_out']
    prob = prob[0,0,:,:]
    np.savetxt(output_file, prob)

def three_sigma_rule(time):
    average_time = np.mean(time)
    sigm = np.std(time)
    upper_bound = average_time + (3 * sigm)
    lower_bound = average_time - (3 * sigm)
    valid_time = []
    for i in range(len(time)):
        if lower_bound <= time[i] <= upper_bound:
            valid_time.append(time[i])
    return valid_time

def calculate_average_time(time):
    average_time = np.mean(time)
    return average_time

def calculate_latency(time):
    time.sort()
    latency = np.median(time)
    return latency

def calculate_fps(pictures, time):
    return pictures / time

def create_result_file(filename):
    if os.path.isfile(filename):
        return
    file = open(filename, 'w')
    head = 'Model;Batch size;Device;IterationCount;Latency;Total time (s);FPS;'
    file.write(head + '\n')
    file.close()

def write_row(filename, net_name, batch_size, number_iter, latency, total_time, fps):
    row = '{};{};CPU;{};{:.3f};{:.3f};{:.3f}'.format(net_name, batch_size, number_iter, 
            latency, total_time, fps)
    file = open(filename, 'a')
    file.write(row + '\n')
    file.close()


def main():
    args = build_argparser().parse_args()
    create_result_file(args.result_file)
    
    # Load network
    net, transformer= load_network(args.proto, args.model, args.batch_size)
    
    # Execute network
    pred, inference_time, total_time = caffe_benchmark(net, transformer, args.number_iter,
                                     args.input_folder, args.output,
                                     args.output_folder, args.task_type, args.batch_size)

    # Write benchmark results
    inference_time = three_sigma_rule(inference_time)
    latency = calculate_latency(inference_time)
    fps = calculate_fps(args.number_iter, total_time)
    write_row(args.result_file, os.path.basename(args.model), args.batch_size,
              args.number_iter, latency, total_time, fps)

if __name__ == '__main__':
    main()