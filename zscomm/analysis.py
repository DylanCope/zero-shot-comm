import tensorflow as tf


def make_map(label, message, num_classes):
    *_, msg_size = tf.shape(message)
    lm_map_shape = (num_classes, msg_size)
    class_indices = [i for i, _ in enumerate(class_labels)]
    indices = tf.reshape(tf.repeat(class_indices, msg_size), 
                         lm_map_shape)
    indices = tf.cast(indices, tf.int64) == label
    return message * tf.cast(indices, tf.float32)


def create_mean_class_message_map(games_played):
    """
    The mean label-message map is a representation of the 
    communication protocol being used for the test messages 
    being sent by the teacher. Each row correponds with a 
    class, each column corresponds with a symbol
    """
    messages = tf.concat([
        history[-1]['message_from_teacher'] 
        for *_, (_, history) in games_played
    ], axis=0)
    
    labels = tf.concat([
        tf.argmax(targets[-1], axis=-1) 
        for _, targets, _ in games_played
    ], axis=0)

    *_, num_classes = tf.shape(games_played[0][1])
    *_, msg_size = tf.shape(messages)
    
    lm_map = tf.zeros((num_classes, msg_size))
    for label, message in zip(labels, messages):
        lm_map = lm_map + make_map(label, message, num_classes)

    row_totals = tf.reduce_sum(lm_map, axis=1)
    row_totals = tf.repeat(row_totals, msg_size)
    row_totals = tf.reshape(row_totals, tf.shape(lm_map))

    return lm_map / row_totals


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