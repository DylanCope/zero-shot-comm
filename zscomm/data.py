import random

import numpy as np
import tensorflow as tf


def generate_batch(source, batch_size=32, num_classes=3):
    inputs = []
    targets = []
    class_labels = list(source.keys())
    
    for batch in range(batch_size):
        batch_tars = []
        batch_inps = []
        
        for ts in range(num_classes):
            sample_space = set(class_labels) - set(batch_tars)
            target = random.choice(list(sample_space))
            batch_tars.append(target)
            inp = random.choice(source[target])
            batch_inps.append(tf.constant(inp))
        
        target = random.choice(list(set(class_labels)))  
        batch_tars.append(target)
        inp = random.choice(source[target])
        batch_inps.append(tf.constant(inp))
        
        batch_tars = [
            tf.one_hot(tar-1, num_classes)
            for tar in batch_tars
        ]
        
        inputs.append(batch_inps)
        targets.append(batch_tars)
    
    inputs = [
        tf.concat([[inputs[b][t]] for b in range(batch_size)], 0)
        for t in range(num_classes + 1)
    ]
    inputs = tf.convert_to_tensor(inputs)
    
    targets = [
        tf.concat([[targets[b][t]] for b in range(batch_size)], 0)
        for t in range(num_classes + 1)
    ]
    targets = tf.convert_to_tensor(targets)
    
    return inputs, targets 


def create_label_to_data_map(data, class_labels):
    return {
        label: [
            img for img, l in data
            if l == label
        ]
        for label in class_labels
    }


def get_mnist_data(num_classes=3):
    class_labels = list(range(num_classes))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def preprocess_images(images):
        images = images / 255.
        return np.where(images > .5, 1.0, 0.0).astype('float32')

    x_train = preprocess_images(x_train)
    train_data = [(x, y) for x, y in zip(x_train, y_train) if y in class_labels]
    np.random.shuffle(train_data)

    x_test = preprocess_images(x_test)
    test_data = [(x, y) for x, y in zip(x_test, y_test) if y in class_labels]
    np.random.shuffle(test_data)
    
    train_data_by_label = create_label_to_data_map(train_data, class_labels)
    test_data_by_label = create_label_to_data_map(test_data, class_labels)
    
    return train_data_by_label, test_data_by_label


def get_simple_card_data(num_classes=3):    
    class_labels = list(range(1, 1 + num_classes))
    sample_size = int(np.ceil(np.log2(num_classes)))
    train_data = [
        ([0.0]*(sample_size - len(bin(l)) + 2) + [float(x) for x in bin(l)[2:]], l)
        for l in range(1, num_classes+1)
    ]
    test_data = train_data
    
    train_data_by_label = create_label_to_data_map(train_data, class_labels)
    test_data_by_label = create_label_to_data_map(test_data, class_labels)
    
    return train_data_by_label, test_data_by_label