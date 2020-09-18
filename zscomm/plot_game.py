from .loss import (
    student_pred_matches_implied_class,
    student_pred_matches_test_class,
    teacher_test_message_is_correct,
    get_expected_student_pred,
    get_correct_teacher_msg
)   

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


def plot_game(inputs, outputs, targets, 
              select_batch=0, 
              class_labels=None,
              show_utterances=True,
              use_mnist=False):
    """
    Assuming that there is a batch of games, this will only plot
    the first one in the batch
    """
    student_preds, history = outputs
    
    if student_preds is not None:
        student_loss = student_pred_matches_implied_class(outputs, targets)
        teacher_loss = student_pred_matches_test_class(outputs, targets)
    else:
        teacher_loss = teacher_only_loss_fn(outputs, targets)
    
    _, _, num_classes = tf.shape(targets)
    num_classes = int(num_classes)
    class_labels = class_labels or \
        list([str(i+1) for i in range(num_classes)])
    
    chan_size, = history[0]['message_from_teacher'][select_batch].shape
    
    if len(history) > 1:
        k = 1.5
        fig = plt.figure(
            figsize=(
                int(k*(num_classes+1)), 
                int(k*(chan_size+2))
            )
        )
        n_rows = len(history) + 1
    else:
        fig = plt.figure(figsize=(10, num_classes+1)) 
        n_rows = 2
    
    gs = gridspec.GridSpec(n_rows, 3) 

    for i, item in enumerate(history):
        ax0 = plt.subplot(gs[i, :2])
        if show_utterances:
            vals = tf.stack([
                item['message_from_teacher'][select_batch],
                tf.nn.softmax(item['teacher_utterance'][select_batch]),
            ])
            vals = tf.reshape(vals, (np.prod(vals.shape),))
            types = ['message'] * chan_size + ['utterance'] * chan_size
            x = list(range(chan_size))
            x = x+x
        else:
            vals = item['message_from_teacher'][select_batch]
            types = ['message'] * chan_size
            x = list(range(chan_size))
        df = pd.DataFrame({
            'x': x, 'y': vals, 'type': types,
        })
        sns.barplot(data=df, x='x', y='y', hue='type',
                    palette=sns.color_palette()[::-1]);
        ax0.set_xlabel(''); ax0.set_ylabel('')
        if i > 0:
            plt.legend([],[], frameon=False)
        else:
            plt.gca().legend().set_title('')
        
        ax0.set_ylim([0, 1])
        ax0.set_yticks([0, 0.5, 1])
        if i < len(history) - 1:
            did_mutate = 1.0 == item['message_mutations'][select_batch][0]
            ax0.set_title(f'Teacher Message {i+1} (Establishing Protocol Phase)\n'
                          f'Did Mutate = {did_mutate}')
        else:
            ax0.set_title(f'Teacher Message {i+1} (Testing Phase)')

        ax1 = plt.subplot(gs[i, 2])
        if len(history) != len(inputs):
            inp = inputs[-1][select_batch]
        else:
            inp = inputs[i][select_batch]
            
        if use_mnist:
            ax1.imshow(inp)        
            ax1.axis('off')
        else:
            domain = list(range(len(inp)))
            ax1.bar(domain, inp)
            ax1.set_ylim([0, 1])
            ax1.set_yticks([0, 1])
            ax1.set_xticks([0, 1])
        
        tar_cls = tf.argmax(targets[i][select_batch])
        ax1.set_title(f'Input (Class {class_labels[tar_cls]})')

    if n_rows > 2:
        ax = plt.subplot(gs[-1, 0])
        sns.barplot(x=list(range(chan_size)),
                    y=get_correct_teacher_msg(history, targets)[select_batch],
                    palette=sns.color_palette()[:1]);
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 0.5, 1])
        
        ax.set_title(f'What the teacher should\n'
                     f'have said in message {i+1}\n'
                     f'(Teacher Loss {int(teacher_loss[select_batch]*10000)/10000})')
            
        ax.set_xlabel('Symbol')
    
    if student_preds is not None:
        ax = plt.subplot(gs[-1, 1])
        df = pd.DataFrame({
            'x': [str(l) for l in class_labels], 
            'y': student_preds[select_batch]
        })
        sns.barplot(data=df, x='x', y='y');
        ax.set_ylim([0, 1])

        ax.set_title(f'What student predicted \n '
                     f'(Student Loss = {int(student_loss[select_batch]*10000)/10000})')
        ax.set_xlabel('Class');  ax.set_ylabel('')

        ax = plt.subplot(gs[-1, 2])
        df = pd.DataFrame({
            'x': [str(l) for l in class_labels], 
            'y': get_expected_student_pred(outputs, targets)[select_batch]
        })
        sns.barplot(data=df, x='x', y='y');
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(f'What the student should\nhave predicted')
        ax.set_xlabel('Class');  ax.set_ylabel('')
    
    plt.tight_layout()
    plt.show()