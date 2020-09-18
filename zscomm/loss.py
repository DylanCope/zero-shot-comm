import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy as cce


def get_sent_messages(history):
    return tf.convert_to_tensor([
        item['message_from_teacher'] for item in history
    ])


def get_comm_protocol(messages):
    return tf.einsum('ijk->jik', messages[:-1])


def get_correct_teacher_msg(history, targets):
    messages = get_sent_messages(history)
    comm_protocol = get_comm_protocol(messages)

    batch_size = tf.shape(comm_protocol)[0]
    
    # labels for each input in the protocol phase
    protocol_labels = tf.argmax(targets[:-1], axis=-1)
    # labels for the final input
    final_labels = tf.argmax(targets[-1], axis=-1)
    final_labels = tf.reshape(final_labels, (1, batch_size)) 
    idx = protocol_labels == final_labels

    correct_msgs = comm_protocol[tf.transpose(idx)]
    return correct_msgs


def get_expected_student_pred(outputs, targets):
    student_preds, history = outputs

    messages = get_sent_messages(history)
    comm_protocol = get_comm_protocol(messages)
    
    batch_size = tf.shape(comm_protocol)[0]
    discretised_protocol = tf.argmax(comm_protocol, axis=-1)
    discretised_final_message = tf.argmax(messages[-1], axis=-1)
    discretised_final_message = tf.reshape(discretised_final_message, 
                                           (batch_size, 1))

    idx = discretised_protocol == discretised_final_message
    idx_num = tf.cast(idx, tf.float32) 
    row_sum = tf.reduce_sum(idx_num, axis=-1)
    row_sum = tf.reshape(row_sum, (batch_size, 1))
    dont_know = row_sum == 0.0

    protocol_labels = tf.transpose(tf.argmax(targets[:-1], axis=-1))
    correct_preds = tf.cast(protocol_labels + 1, tf.float32) * idx_num
    correct_preds = tf.cast(correct_preds, tf.int32)
    num_classes = tf.shape(targets)[2]
    correct_preds = tf.one_hot(correct_preds - 1, num_classes, axis=-1)

    pred_size = tf.shape(correct_preds)[2]
    uniform_preds = tf.ones((batch_size, pred_size)) 
    uniform_preds = uniform_preds * tf.cast(dont_know, tf.float32)
    uniform_preds = tf.reshape(uniform_preds, (batch_size, 1, pred_size))

    correct_preds = tf.concat([correct_preds, uniform_preds], axis=-2)
    correct_preds = tf.reduce_sum(correct_preds, axis=-2)

    row_sums = tf.reduce_sum(correct_preds, axis=-1)
    row_sums = tf.reshape(row_sums, (batch_size, 1))
    correct_preds = correct_preds / row_sums
    
    return correct_preds


def student_pred_matches_test_class(outputs, targets):
    student_preds, history = outputs
    return cce(targets[-1], student_preds)
    

def student_pred_matches_implied_class(outputs, targets):
    """ Implied class according to teacher message """
    student_preds, history = outputs
    expected_preds = get_expected_student_pred(outputs, targets)
    return cce(expected_preds, student_preds)


def teacher_only_loss_fn(outputs, targets):
    _, history = outputs
    correct_msg = get_correct_teacher_msg(history, targets)
    teacher_utt = tf.nn.softmax(history[-1]['teacher_utterance'])
    return cce(correct_msg, teacher_utt)


def combined_loss_function(outputs, targets):
    loss_s = student_pred_matches_implied_class(outputs, targets)
    loss_t = student_pred_matches_test_class(outputs, targets)
    return loss_s + loss_t
