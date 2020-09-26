from .agent import Agent
from .comm_channel import CommChannel

import random

import numpy as np
import tensorflow as tf


def maybe_mutate_message(
    message, history, p_mutate, 
    is_kind=True, training=False
):
    batch_size = tf.shape(message)[0]
    channel_size = tf.shape(message)[1]
        
    if not is_kind:
        history = []

    if history == []:
        history = [{'message_from_teacher': tf.zeros_like(message)}]

    possible_utts = np.array(range(channel_size))

    prev_utts = tf.convert_to_tensor([
        tf.argmax(item['message_from_teacher'], axis=-1) 
        for item in history 
    ])
    prev_utts = tf.transpose(prev_utts)

    possible_utts = tf.convert_to_tensor([list(range(channel_size))], 
                                         dtype=tf.int64)
    possible_utts = tf.repeat(possible_utts, len(history), axis=0)
    possible_utts = tf.repeat([possible_utts], batch_size, axis=0)

    idx = possible_utts != tf.reshape(tf.repeat(prev_utts, channel_size, axis=-1), 
                                      possible_utts.shape)
    idx = tf.reduce_sum(tf.cast(idx, tf.int64), axis=-2) == len(history)
    num_remaining_utts = channel_size - len(history)

    noise = tf.random.uniform(idx.shape) * tf.cast(idx, tf.float32)
    random_choice = tf.repeat(tf.argmax(noise, axis=1), channel_size)
    random_choice = tf.reshape(random_choice, (batch_size, channel_size))
    random_choice_idx = random_choice == possible_utts[:, 0, :]
    
    possible_mutations = possible_utts[:, 0, :][random_choice_idx]
    possible_mutations = tf.one_hot(possible_mutations, channel_size)
    possible_mutations = tf.stop_gradient(possible_mutations)

    rand_samples = tf.random.uniform((batch_size, 1))
    mask = tf.cast(rand_samples < p_mutate, tf.float32)

    mutated_message = mask * possible_mutations + (1 - mask) * message

    return mutated_message, mask


def create_permutation_map(batch_size, channel_size):

    indices_col = []
    for i in range(batch_size):
        idxs = list(range(channel_size))
        random.shuffle(idxs)
        indices_col.append(idxs)
    indices_col = tf.convert_to_tensor(indices_col)

    indices_row = tf.repeat([list(range(batch_size))], channel_size, axis=0)
    indices_row = tf.transpose(indices_row)
    
    permutation_map = tf.stack([indices_row, indices_col], axis=-1)
    
    return permutation_map


def apply_permutation(permutation_map, message):
    return tf.gather_nd(message, permutation_map)
    
    
def play_game(
    inputs, teacher, student, 
    comm_channel=None, 
    p_mutate=0.5,
    kind_mutations=True,
    message_permutation=False,
    channel_size=5,
    channel_temp=1,
    channel_noise=0.5,
    access_to_inputs_in_first_phase=True,
    stop_gradients_on_final_message=False,
    stop_gradients_on_all_comm=False,
    no_protocol_establishment=False,
    training=False,
):
    comm_channel = comm_channel or \
        CommChannel(size=channel_size, 
                    temperature=channel_temp, 
                    noise=channel_noise)

    num_inputs = tf.shape(inputs)[0]
    batch_size = tf.shape(inputs)[1]
    
    if message_permutation:
        permutation_map = create_permutation_map(batch_size, 
                                                 comm_channel.size)
    else:
        permutation_map = None

    no_inp = tf.zeros_like(inputs[0])
    silence = comm_channel.get_initial_state(batch_size)

    teacher_prev_msg = silence
    history = []
    
    teacher_state = None
    student_state = None

    if not no_protocol_establishment:
        for i in range(num_inputs - 1):
            inp = inputs[i]
            
            if access_to_inputs_in_first_phase:
                teacher_inputs = (inp, teacher_prev_msg, silence)
            else:
                teacher_inputs = (no_inp, teacher_prev_msg, silence)

            
            teacher_utterance, _, teacher_state = teacher(
                teacher_inputs, state=teacher_state, training=training
            )

            message_from_teacher = comm_channel(teacher_utterance, 
                                                training=training)
            message_from_teacher, mutations = \
                maybe_mutate_message(message_from_teacher, 
#                                      comm_channel.size,
                                     history,
                                     p_mutate,
                                     is_kind=kind_mutations,
                                     training=training)
            
            if message_permutation:
                message_from_teacher = apply_permutation(permutation_map, 
                                                         message_from_teacher)
                     
            if stop_gradients_on_all_comm:
                message_from_teacher = tf.stop_gradient(message_from_teacher)

            if access_to_inputs_in_first_phase:
                student_inputs = (inp, silence, message_from_teacher)
            else:
                student_inputs = (no_inp, silence, message_from_teacher)

            _, _, student_state = student(
                student_inputs, state=student_state, training=training
            )

            history.append({
                'teacher_utterance': teacher_utterance,
                'message_from_teacher': message_from_teacher, 
                'message_mutations': mutations,
                'permutation_map': permutation_map
            })
            teacher_prev_msg = message_from_teacher

    teacher_inputs = (inputs[-1], teacher_prev_msg, silence)
    teacher_utterance, _, _ = teacher(
        teacher_inputs, state=teacher_state, training=training
    )

    message_from_teacher = comm_channel(teacher_utterance,
                                        training=training)

    if stop_gradients_on_final_message or stop_gradients_on_all_comm:
        message_from_teacher = tf.stop_gradient(message_from_teacher)

    student_inputs = (no_inp, silence, message_from_teacher)
    _, student_preds, _ = student(
        student_inputs, state=student_state, training=training
    )

    history.append({
        'teacher_utterance': teacher_utterance,
        'message_from_teacher': message_from_teacher, 
        'message_mutations': None
    })

    return student_preds, history