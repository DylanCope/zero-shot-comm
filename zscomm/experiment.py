from IPython.display import clear_output
import unittest.mock as mock
import time

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf

from .synth_teacher import SyntheticTeacher
from .loss import *
from .play_game import play_game


class Experiment:
    
    def __init__(
        self, 
        train_data_gen_fn,
        test_data_gen_fn,
        play_params=None,
        teacher=None, 
        student=None,
        loss_fn=complete_loss_fn,
        loss_kwargs=None,
        student_loss_fn=None,
        teacher_loss_fn=None,
        max_epochs=40, 
        steps_per_epoch=50, 
        step_print_freq=5,
        test_freq=5,
        test_steps=25,
        lr=1e-2, # learning rate
        print_prehistory=None,
        name='experiment'
    ):
        self.generate_train_batch = train_data_gen_fn
        self.generate_test_batch = test_data_gen_fn
        self.play_params = play_params or dict()
        self.student = student
        self.teacher = teacher
            
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs or dict()
        
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.step_print_freq = step_print_freq
        self.test_freq = test_freq
        self.test_steps = test_steps
        
        self.training_history = []
        self.epoch = 0
        self.optimiser_1 = tf.keras.optimizers.RMSprop(learning_rate=lr)
        self.optimiser_2 = tf.keras.optimizers.RMSprop(learning_rate=lr)
        
        self.optimise_separately = (student_loss_fn is not None) or \
                                   (teacher_loss_fn is not None)
        self.student_loss_fn = student_loss_fn or loss_fn
        self.teacher_loss_fn = teacher_loss_fn or loss_fn
        
        self.print_prehistory = print_prehistory or (lambda: None)
        self.name = name
        
        self.results = None
        
    def get_trainable_variables(self):
        if self.student is not None and self.teacher is None:
            return self.student.trainable_variables
        elif self.student is None and self.teacher is not None:
            return self.teacher.trainable_variables
        elif self.student == self.teacher:
            return self.student.trainable_variables
        else:
            return self.student.trainable_variables + \
                   self.teacher.trainable_variables
        
    def get_play_params(self):
        if isinstance(self.play_params, dict):
            return self.play_params
        # otherwise assume play_params is a function
        return self.play_params(self.epoch)

    def training_step(self):
        inputs, targets = self.generate_train_batch()
        
        student = self.student or \
            mock.MagicMock(return_value=(None, None, None))
        
        teacher = self.teacher or \
            SyntheticTeacher(CHANNEL_SIZE, NUM_CLASSES, targets)
    
        with tf.GradientTape(persistent=True) as tape:
            outputs = play_game(
                inputs, teacher, student, 
                training=True, 
                **self.get_play_params()
            )

            if self.optimise_separately:
                loss_s = self.student_loss_fn(outputs, targets)
                loss_t = self.teacher_loss_fn(outputs, targets)
                loss = loss_s + loss_t
            else:
                loss = self.loss_fn(outputs, targets)

        if self.optimise_separately:
            trainable_vars_s = self.student.trainable_variables
            grads_s = tape.gradient(loss_s, trainable_vars_s)
            self.optimiser_1.apply_gradients(zip(grads_s, trainable_vars_s))
            
            trainable_vars_t = self.teacher.trainable_variables
            grads_t = tape.gradient(loss_t, trainable_vars_t)
            self.optimiser_2.apply_gradients(zip(grads_t, trainable_vars_t))
        else:
            trainable_vars = self.get_trainable_variables()
            grads = tape.gradient(loss, trainable_vars)
            self.optimiser_1.apply_gradients(zip(grads, trainable_vars))
    
        return loss
    
    def get_test_loss(self, games_played):
        test_loss = tf.reduce_mean([
            self.loss_fn(outputs, targets, **self.loss_kwargs)
            for _, targets, outputs in games_played
        ])
        return float(test_loss.numpy())
    
    def get_student_test_metrics(self, games_played):
        
        ground_truth_labels = tf.concat([
            tf.argmax(targets[-1], axis=-1) 
            for _, targets, _ in games_played
        ], axis=0)

        preds = tf.concat([
            tf.argmax(student_preds, axis=-1) 
            for _, _, (student_preds, _) in games_played
        ], axis=0)
        
        correct_preds = [
            get_expected_student_pred(outputs, targets)
            for (_, targets, outputs) in games_played
        ]
        
        mean_ground_truth_f1 = f1_score(
            ground_truth_labels.numpy(), preds.numpy(), 
            average='micro'
        )
        mean_ground_truth_f1 = float(mean_ground_truth_f1)
        
        ground_truth_acc = accuracy_score(
            ground_truth_labels.numpy(), preds.numpy(),
        )
        ground_truth_acc = float(ground_truth_acc)
        
        mean_student_error = tf.reduce_mean([
            student_pred_matches_implied_class(outputs, targets)
            for _, targets, outputs in games_played
        ])
        mean_student_error = float(mean_student_error.numpy().mean())
        
        return {
            'mean_ground_truth_f1': mean_ground_truth_f1,
            'ground_truth_acc': ground_truth_acc,
            'mean_student_error': mean_student_error,
        }
    
    def get_teacher_test_metrics(self, games_played):
        
        mean_teacher_error = tf.reduce_mean([
            teacher_test_message_is_correct(outputs, targets)
            for _, targets, outputs in games_played
        ])
        mean_teacher_error = float(mean_teacher_error.numpy().mean())
        
        mean_protocol_diversity = tf.reduce_mean([
            protocol_diversity_loss(outputs)**-1
            for _, targets, outputs in games_played
        ])
        mean_protocol_diversity = \
            float(mean_protocol_diversity.numpy().mean())
        
        return {
            'mean_teacher_error': mean_teacher_error,
            'mean_protocol_diversity': mean_protocol_diversity,
        }
    
    def extract_test_metrics(self, games_played):
        
        student_metrics = \
            self.get_student_test_metrics(games_played)
        
        teacher_metrics = \
            self.get_teacher_test_metrics(games_played)

        return {
            'mean_test_loss': self.get_test_loss(games_played), 
            **student_metrics, **teacher_metrics
        }
    
    def test_play(self, inputs):
        
        student = self.student or \
            mock.MagicMock(return_value=(None, None, None))
        teacher = self.teacher or \
            SyntheticTeacher(self.channel_size, 
                             self.num_classes, 
                             targets)
        
        return play_game(inputs, teacher, student, 
                         training=False,
                         **self.get_play_params())
        
    def run_tests(self):
        
        test_samples = [
            self.generate_test_batch()
            for _ in range(self.test_steps)
        ]

        games_played = [
            (inp, tar, self.test_play(inp))
            for inp, tar in test_samples
        ]

        test_metrics = self.extract_test_metrics(games_played)
        
        return games_played, test_metrics
    
    def run_training_epoch(self):
        mean_loss = None

        start_time = time.time()
        for step in range(self.steps_per_epoch):

            loss = self.training_step()
            if mean_loss is None:
                mean_loss = tf.reduce_mean(loss)
            else:
                mean_loss = (mean_loss + tf.reduce_mean(loss)) / 2.0

            if step % self.step_print_freq == 0:
                self.print_history()
                self.print_step_progress(step, mean_loss)
                clear_output(wait=True)

        seconds_taken = time.time() - start_time
        self.training_history.append({
            'loss': float(mean_loss.numpy().mean()), 
            'seconds_taken': seconds_taken,
            'experiment_state': self.get_config()
        })
        
    def _test_step(self):
        self.print_history()
        print('Running test games...')
        clear_output(wait=True)
        _, test_metrics = self.run_tests()
        self.training_history[-1]['test_metrics'] = test_metrics
        return test_metrics
        
    def _run_internal(self):
        while self.epoch < self.max_epochs:
            self.run_training_epoch()

            if self.epoch % self.test_freq == 0:
                self._test_step()

            self.epoch += 1
            self.print_history()
            clear_output(wait=True)

        self.results = self._test_step()
    
    def run(self, catch_interrupt=True):
        self.print_history()
        clear_output(wait=True)
        if catch_interrupt:
            try:
                self._run_internal()
            except KeyboardInterrupt:
                pass
        else:
            self._run_internal()

        self.print_history()
        print('Training stopped.')
    
    def get_config(self):
        optimiser_config = {
            k: v if not isinstance(v, np.float32) else float(v)
            for k, v in self.optimiser_1.get_config().items()
        }
        if self.optimise_separately:
            loss_config = {
                'teacher_loss_fn': self.teacher_loss_fn.__name__,
                'student_loss_fn': self.student_loss_fn.__name__,
            }
        else:
            loss_config = {
                'loss_fn': self.loss_fn.__name__,
            }
        return {
            'name': self.name,
            'max_epochs': self.max_epochs,
            'steps_per_epoch': self.steps_per_epoch,
            'epochs_optimised': self.epoch,
            'play_params': self.get_play_params(),
            'test_freq': self.test_freq,
            'test_steps': self.test_steps,
            'optimiser_config': optimiser_config,
            'optimise_agents_separately': self.optimise_separately,
            **loss_config
        }
    
    def print_test_metrics(self, metrics):
        print(
            f"Test Loss: {round(metrics['mean_test_loss'], 3)},",
            f"Ground Truth F1-Score: {round(metrics['mean_ground_truth_f1'], 3)},",
            f"Student Error: {round(metrics['mean_student_error'], 3)},",
            f"Teacher Error: {round(metrics['mean_teacher_error'], 3)},",
            f"Protocol Diversity: {round(metrics['mean_protocol_diversity'], 3)},"
        )
    
    def print_results(self):
        self.print_test_metrics(self.results)

    def print_history(self):
        self.print_prehistory()
        print(f'Running {self.name}...')
        print('Run config:\n', self.get_config())
        for e, item in enumerate(self.training_history):
            mins = int(item['seconds_taken']) // 60
            secs = int(item['seconds_taken']) % 60
            loss = round(item['loss'], 3)
            print(f'Epoch {e}, Time Taken (mm:ss): {mins}:{secs}, Mean Loss: {loss}')
            if 'test_metrics' in item:
                self.print_test_metrics(item['test_metrics'])

    def print_step_progress(self, step, step_mean_loss):
        l = round(float(step_mean_loss.numpy().mean()), 4)
        p = round(100 * step / self.steps_per_epoch, 2)
        print(f'Epoch {self.epoch}, {p}% complete, Loss: {l}')
        
    def plot_training_history(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            ax = axs[0]
            ax.set_title('Loss History')
            ax = axs[1]
            ax.set_title('Test Metrics History')

        ax = axs[0]
        sns.lineplot(x=range(len(self.training_history)), 
                     y=[item['loss'] for item in self.training_history],
                     label='train_loss_exp_1',
                     ax=ax);

        test_metric_items = [
            item['test_metrics'] 
            for item in self.training_history
            if 'test_metrics' in item
        ]
        epochs = [
            epoch * self.test_freq 
            for epoch, item in enumerate(test_metric_items)
        ]
        metrics = list(test_metric_items[0].keys())

        for metric in metrics:
            ax = axs[1] if metric != 'mean_test_loss' else axs[0]
            sns.lineplot(x=epochs, 
                         y=[item[metric] for item in test_metric_items],
                         label=f'{metric}_{self.name}',
                         ax=ax)
        return axs