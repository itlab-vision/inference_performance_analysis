# -*- coding: utf-8 -*-
"""
@author: Evgenii Vasiliev

OpenCV classification benchmarking script 


Sample string to run benchmark: 

cd IXPUG_DNN_benchmark/opencv_benchmark

mkdir results_classification
python3 opencv_benchmark.py -i ../datasets/imagenet/ -p ../models/resnet-50.prototxt -m ../models/resnet-50.caffemodel -ni 1000 -of ./results_classification/ -r ./results_classification/result.csv -w 224 -he 224 -s 1.0



Last modified 25.07.2019

"""

import cv2
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
    parser.add_argument('-w', '--width', help='Input tensor width', 
        required=True, type=int)
    parser.add_argument('-he', '--height', help='Input tensor height', 
        required=True, type=int)
    parser.add_argument('-s', '--scale', help='Input tensor values scaling', 
        required=True, type=float)
    parser.add_argument('-i', '--input_folder', help='Name of input folder',
        default='', type=str)
    parser.add_argument('-ni', '--number_iter', help='Number of inference \
        iterations', required=True, type=int)
    parser.add_argument('-o', '--output', help='Get output',
        default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-of', '--output_folder', help='Name of output folder',
        default='', type=str)
    parser.add_argument('-r', '--result_file', help='Name of output folder', 
        default='result.csv', type=str)
    parser.add_argument('-t', '--task_type', help='Task type: \
        classification / detection', default = 'classification', type=str)
    parser.add_argument('-b', '--batch_size', help='batch size', 
        required=True, type=int)
        
    return parser

def load_network(model, config):
    net = cv2.dnn.readNet(model, config)
    return net

def load_images(w, h, input_folder, numbers):
    data = os.listdir(input_folder)
    counts = numbers
    if len(data)<numbers:
        counts = len(data)
    images = []
    for i in range(counts):
        image = cv2.imread(os.path.join(input_folder, data[i]))
        image = cv2.resize(image, (w, h))
        images.append(image)
        del image
    return images, counts

def opencv_benchmark(net, number_iter, input_folder, 
                    need_output = False, output_folder = '', task_type = '', 
                    blob_size = (224,224), blob_scale = 1.0, batch_size = 1):
    
    filenames = os.listdir(input_folder)
    inference_time = []
    
    number_iter = (number_iter + batch_size - 1) // batch_size
    images, counts = load_images(blob_size[0], blob_size[1], input_folder, number_iter * batch_size)
    
    t0_total = time()
    for i in range(number_iter):
        if batch_size > 1:
            a = (i * batch_size) % len(images) 
            b = (((i + 1) * batch_size - 1) % len(images)) + 1
            im_batch = images[a : b]
            blob = cv2.dnn.blobFromImages(im_batch, blob_scale, blob_size, (0, 0, 0))
        else:
            blob = cv2.dnn.blobFromImage(images[i], blob_scale, blob_size, (0, 0, 0))
        
        net.setInput(blob)
        
        
        t0 = time()
        preds = net.forward()
        t1 = time()
        
        if (need_output == True):
            
            if batch_size == 1:
                # Generate output name
                output_filename = str(os.path.splitext(os.path.basename(image_name))[0])+'.npy'
                output_filename = os.path.join(os.path.dirname(output_folder), output_filename) 
                # Save output
                if task_type == 'classification':
                    classification_output(preds, output_filename)
                elif task_type == 'detection':
                    detection_output(preds, output_filename)
                
        inference_time.append(t1 - t0)
    t1_total = time()
    inference_total_time = t1_total - t0_total
    return preds, inference_time, inference_total_time

def classification_output(prob, output_file):
    prob = prob[0]
    np.savetxt(output_file, prob)

def detection_output(prob, output_file):
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

def write_row(filename, net_name, number_iter, batch_size, latency, total_time, fps):
    row = '{};{};CPU;{};{:.3f};{:.3f};{:.3f}'.format(net_name, batch_size, number_iter, 
           latency, total_time, fps)
    file = open(filename, 'a')
    file.write(row + '\n')
    file.close()

def main():
    args = build_argparser().parse_args()
    create_result_file(args.result_file)
    
    
    # Load network
    net = load_network(args.model, args.proto)
    
    # Execute network
    pred, inference_times, total_time = opencv_benchmark(net, args.number_iter,
                                     args.input_folder, args.output,
                                     args.output_folder,args.task_type,
                                     (args.width, args.height), args.scale, args.batch_size)
    
    # Write benchmark results
    inference_times = three_sigma_rule(inference_times)
    latency = calculate_latency(inference_times)
    fps = calculate_fps(args.number_iter, total_time)
    write_row(args.result_file, os.path.basename(args.model), args.number_iter, 
              args.batch_size, latency, total_time, fps)

if __name__ == '__main__':
    main()
