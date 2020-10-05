from IPython.display import clear_output
from itertools import combinations
import json
from pathlib import Path
import uuid

from .play_game import play_game


def test_game(teacher, student, generate_test_batch, num_tests=5, **zs_play_kwargs):
    games_played = []
    for _ in range(num_tests):
        inputs, targets = generate_test_batch()
        outputs = play_game(
            inputs, teacher, student, 
            training=False, 
            **zs_play_kwargs
        )
        games_played.append((inputs, targets, outputs))
    return games_played


def write_json(obj, loc):
    with Path(loc).open(mode='w') as f:
        json.dump(obj, f)


def export_experiment(experiment, location):
    Path(location).mkdir()
    
    write_json(experiment.training_history,
               f'{location}/training_history.json')
    
    write_json(experiment.get_config(),
               f'{location}/config.json')
    
    write_json(experiment.results,
               f'{location}/results.json')
    
    experiment.teacher.save_weights(f'{location}/agent_weights')
    
    print('Experiment saved at:', location)
    
    
def measure_zero_shot_coordination(experiment_1,
                                   experiment_2,
                                   num_tests=5, 
                                   **zs_play_kwargs):
    
    results = []
    games_played = test_game(experiment_1.teacher, 
                             experiment_2.student,
                             experiment_1.generate_test_batch,
                             num_tests=num_tests,
                             **zs_play_kwargs)
    test_metrics = experiment_1.extract_test_metrics(games_played)
    results.append(test_metrics)

    games_played = test_game(experiment_2.teacher, 
                             experiment_1.student,
                             experiment_1.generate_test_batch,
                             num_tests=num_tests,
                             **zs_play_kwargs)
    test_metrics = experiment_1.extract_test_metrics(games_played)
    results.append(test_metrics)
    
    return results


class MetaExperiment:
    
    def __init__(
        self,
        create_experiment_fn,
        num_experiments=3,
        print_prehistory=None,
        name='meta_experiment',
        export_location=None,
        **make_experiment_kwargs
    ):
        self.num_experiments = num_experiments
        self.results = None
        self.print_prehistory = print_prehistory or (lambda: None)
        self.name = name
        
        self.export_location = export_location
        if export_location is not None:
            Path(export_location).mkdir(exist_ok=True)
        
        self.experiments = [
            {
                'experiment': create_experiment_fn(
                    print_prehistory=self.print_history,
                    **make_experiment_kwargs
                ),
                'status': 'Not Run',
                'results': None,
                'index': i,
            }
            for i in range(num_experiments)
        ]
    
    def print_history(self):
        self.print_prehistory()
        print(f'Running {self.name}...')
        num_complete = len([
            item for item in self.experiments
            if item['status'] == 'Complete'
        ])
        for item in self.experiments:
            if item['status'] == 'Complete':
                item['experiment'].print_results()
        for item in self.experiments:
            if item['status'] == 'In Progress':
                print(f"Running experiment {item['index']}", 
                      f'({num_complete}/{self.num_experiments} complete):')
                break
                
    def print_results(self):
        print(self.name, 'results: ', self.results)

    def is_finished(self):
        return all([
            item['status'] == 'Complete' 
            for item in self.experiments
        ])
    
    def get_experiment_to_run(self):
        for item in self.experiments:
            if item['status'] == 'In Progress':
                return item
        not_run = [
            item
            for item in self.experiments
            if item['status'] == 'Not Run'
        ]
        if len(not_run) == 0:
            return None
        return not_run[0]
    
    def get_experiment_results(self, experiment):
        test_metrics_items = [
            item['test_metrics']
            for item in experiment.training_history
            if 'test_metrics' in item
        ]
        return test_metrics_items[-1]
    
    def _run_next_subexperiment(self):
        experiment_item = self.get_experiment_to_run()
        index = experiment_item['index']
        experiment = experiment_item['experiment']

        self.experiments[index]['status'] = 'In Progress'

        experiment.run(catch_interrupt=False)

        self.experiments[index]['results'] = \
            self.get_experiment_results(experiment)
        
        self.export_experiment(experiment)
        
        self.experiments[index]['status'] = 'Complete'
        
    def export_experiment(self, experiment):
        if self.export_location is not None:
            folder = str(uuid.uuid4())
            location = f'{self.export_location}/{folder}'
            export_experiment(experiment, location)
    
    def _get_results(self):
        return self.measure_zero_shot_coordination()
    
    def _run_internal(self):
        while not self.is_finished():
            self._run_next_subexperiment()
        
        self.results = self._get_results()

    def measure_zero_shot_coordination(self):
        results = []

        for item_1, item_2 in combinations(self.experiments, 2):
            e1 = item_1['experiment']
            e2 = item_2['experiment']
            
            test_metrics = measure_zero_shot_coordination(e1, e2)
            
            results.extend([
                metric['mean_ground_truth_f1']
                for metric in test_metrics
            ])

        return results
    
    def run(self, catch_interrupt=True):
        if catch_interrupt:
            try:
                self._run_internal()
            except KeyboardInterrupt:
                pass
        else:
            self._run_internal()
        
        clear_output()
        self.print_history()
        print('Run Stopped.')