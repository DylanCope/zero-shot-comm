import numpy as np
import tensorflow as tf


def make_map(label, message, num_rows):
    *_, msg_size = tf.shape(message)
    map_shape = (num_rows, msg_size)
    label_indices = list(range(num_rows))
    indices = tf.reshape(tf.repeat(label_indices, msg_size), 
                         map_shape)
    indices = tf.cast(indices, tf.int64) == label
    return message * tf.cast(indices, tf.float32)


def create_mean_class_message_map(games_played):
    """
    The mean class-message map is a representation of the 
    communication protocol being used for the test messages 
    being sent by the teacher. Each row correponds with a 
    class, each column corresponds with a symbol
    """
    messages = tf.concat([
        history[-1]['message_from_teacher'] 
        for *_, (_, history) in games_played
    ], axis=0)
    
    cls_labels = tf.concat([
        tf.argmax(targets[-1], axis=-1) 
        for _, targets, _ in games_played
    ], axis=0)

    *_, num_classes = tf.shape(games_played[0][1])
    *_, msg_size = tf.shape(messages)
    
    cm_map = tf.zeros((num_classes, msg_size))
    for cls_label, message in zip(cls_labels, messages):
        cm_map = cm_map + make_map(cls_label, message, num_classes)

    row_totals = tf.reduce_sum(cm_map, axis=1)
    row_totals = tf.repeat(row_totals, msg_size)
    row_totals = tf.reshape(row_totals, tf.shape(cm_map))

    return cm_map / row_totals


def create_mean_index_message_map(games_played):
    """
    The mean index-message map is a representation of the 
    intraepisodic communication protocol being generated 
    by the teacher. Each row correponds with a 
    time step index, each column corresponds with a symbol
    """
    messages = tf.concat([
        [item['message_from_teacher'] for item in history[:-1]]
        for *_, (_, history) in games_played
    ], axis=0)
    num_ts, batch_size, chan_size = tf.shape(messages)
    messages = tf.reshape(messages, (num_ts*batch_size, chan_size))

    time_step_indices = tf.concat([
        [int(batch_size) * [i] for i, _ in enumerate(history[:-1])]
        for *_, (_, history) in games_played
    ], axis=0)
    time_step_indices = tf.reshape(time_step_indices, (num_ts*batch_size,))
    time_step_indices = tf.cast(time_step_indices, tf.int64)

    num_indices = tf.reduce_max(time_step_indices) + 1
    *_, msg_size = tf.shape(messages)
    
    im_map = tf.zeros((num_indices, msg_size))
    for idx, message in zip(time_step_indices, messages):
        im_map = im_map + make_map(idx, message, num_indices)

    row_totals = tf.reduce_sum(im_map, axis=1)
    row_totals = tf.repeat(row_totals, msg_size)
    row_totals = tf.reshape(row_totals, tf.shape(im_map))

    return im_map / row_totals


def compute_confusion_matrix(games_played):
    labels = tf.concat([
        tf.argmax(targets[-1], axis=-1) 
        for _, targets, _ in games_played
    ], axis=0)

    preds = tf.concat([
        tf.argmax(student_preds, axis=-1) 
        for _, _, (student_preds, _) in games_played
    ], axis=0)

    conf_matrix = tf.math.confusion_matrix(labels, preds)

    col_totals = tf.reduce_sum(conf_matrix, axis=0)
    col_totals = tf.repeat(col_totals, tf.shape(conf_matrix)[0])
    col_totals = tf.reshape(col_totals, tf.shape(conf_matrix))
    col_totals = tf.transpose(col_totals)

    conf_matrix = (conf_matrix / col_totals).numpy()
    conf_matrix[np.where(np.isnan(conf_matrix))] = 0
    
    return conf_matrix