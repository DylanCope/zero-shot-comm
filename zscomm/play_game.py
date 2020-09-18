from .agent import Agent
from .comm_channel import CommChannel

import numpy as np
import tensorflow as tf


def maybe_mutate_message(
    message, channel_size, history, p_mutate, 
    training=False
):
    batch_size = tf.shape(message)[0]

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

    idx = possible_utts != tf.reshape(tf.repeat(prev_utts, 5, axis=-1), 
                                      possible_utts.shape)
    idx = tf.reduce_sum(tf.cast(idx, tf.int64), axis=-2) == len(history)
    num_remaining_utts = channel_size - len(history)

    noise = tf.random.uniform(idx.shape) * tf.cast(idx, tf.float32)
    noise_max = tf.reduce_max(noise, axis=-1)
    random_choice = noise == tf.reshape(noise_max, (batch_size, 1))

    possible_mutations = possible_utts[:, 0, :][random_choice]
    possible_mutations = tf.one_hot(possible_mutations, channel_size)
    possible_mutations = tf.stop_gradient(possible_mutations)

    rand_samples = tf.random.uniform((batch_size, 1))
    mask = tf.cast(rand_samples < p_mutate, tf.float32)

    mutated_message = mask * possible_mutations + (1 - mask) * message
    
    return mutated_message, mask
    
    
def play_game(
    inputs, teacher, student, 
    comm_channel=None, 
    p_mutate=0.5, 
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
                                     comm_channel.size,
                                     history,
                                     p_mutate,
                                     training=training)
                     
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
                'message_mutations': mutations
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