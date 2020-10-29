# -*- coding: utf-8 -*-
"""
@author: Evgenii Vasiliev

TensorFlow benchmarking script

Sample string to run benchmark: 

cd IXPUG_DNN_benchmark/tf_benchmark
mkdir results_classification
mkdir results_detection
python3 tf_benchmark.py -t classification -i ../datasets/imagenet/ -m ../models/resnet-50.pb -ni 1000 -o False -of ./results_classification/ -r ./results_classification/result.csv -w 224 -he 224 -b 1
python3 tf_benchmark.py -t detection -i ../datasets/pascal_voc/ -m ../models/ssd300.pb -ni 1000 -o False -of ./results_detection/ -r ./results_detection/result.csv -w 224 -h 224 -b 1

Created 04.10.2019

"""

import cv2
import tensorflow as tf
import os.path
import argparse
import numpy as np
from time import time

def build_argparser():
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an TensorFlow .pb model\
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
        default=1, type=int)
    parser.add_argument('-w', '--width', help='Input tensor width', 
        required=True, type=int)
    parser.add_argument('-he', '--height', help='Input tensor height', 
        required=True, type=int)
    return parser
	
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

def load_network(model, width, height):
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph = graph)
    
    # Import the TF graph
    with tf.gfile.GFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
                 
    # Define input tensor
    #input_tensor = tf.placeholder(np.float32, shape = [None, width, height, 3], name='input') 
    
    input_tensor = tf.placeholder(tf.uint8, shape=[None, 300, 300, 3],name='input')
    
    tf.import_graph_def(graph_def, {'input': input_tensor})
            
    return sess, graph, input_tensor

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

def tf_benchmark(sess, graph, input_tensor, width, height, number_iter, 
                 input_folder, need_output = False, output_folder = '', 
                 task_type = '', batch_size = 1):
    filenames = os.listdir(input_folder)
    inference_times = []
    number_iter = (number_iter + batch_size - 1) // batch_size
    images, counts = load_images(width, height, input_folder, number_iter * batch_size)
 
    output_tensor = graph.get_tensor_by_name("import/predict:0")
    
    #Warmup
    blob = np.array(images[0:1])
    output = sess.run(output_tensor, feed_dict = {input_tensor: blob})
    
    t0_total = time()
    for i in range(number_iter):
        a = (i * batch_size) % len(images) 
        b = (((i + 1) * batch_size - 1) % len(images)) + 1
        blob = np.array(images[a:b])
        t0 = time()
        output = sess.run(output_tensor, feed_dict = {input_tensor: blob})
        t1 = time()
    
        if (need_output == True and batch_size == 1):
            # Generate output name
            output_filename = str(os.path.splitext(os.path.basename(filenames[i]))[0])+'.npy'
            output_filename = os.path.join(os.path.dirname(output_folder), output_filename) 
            # Save output
            np.savetxt(output_filename, output)
    
        inference_times.append(t1 - t0)
    t1_total = time()
    inference_total_time = t1_total - t0_total
    return inference_times, inference_total_time

def main():	
    args = build_argparser().parse_args()
    create_result_file(args.result_file)

    # Load network	
    sess, graph, input_tensor = load_network(args.model, args.width, args.height)
    
    # Execute network
    inference_time, total_time = tf_benchmark(sess, graph, input_tensor, 
        args.width, args.height, args.number_iter, args.input_folder, args.output, args.output_folder, 
        args.task_type, args.batch_size)

    # Write benchmark results
    inference_time = three_sigma_rule(inference_time)
    latency = calculate_latency(inference_time)
    fps = calculate_fps(args.number_iter, total_time)
    write_row(args.result_file, os.path.basename(args.model), args.batch_size,
              args.number_iter, latency, total_time, fps)

if __name__ == '__main__':
    main()