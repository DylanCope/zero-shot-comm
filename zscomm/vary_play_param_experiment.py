from pathlib import Path
import json

import tensorflow as tf

from .loss import *
from .meta_experiment import MetaExperiment


class VaryPlayParamExperiment(MetaExperiment):
    
    def __init__(self, 
                 param_vals = None,
                 param_name = 'p_mutate',
                 save_location=None, 
                 num_experiments_per_val=3,
                 name='vary_pm_experiment',
                 **experiment_kwargs):
        self.name = name        
        self.param_name = param_name
        self.num_experiments = len(param_vals)
        
        self.experiments = [
            {
                param_name: val,
                'experiment': MetaExperiment(
                    print_prehistory=self.print_history,
                    name=f'meta_experiment_{param_name}={val}',
                    export_location=None if save_location is None else \
                                    f'{save_location}/{param_name}={val}',
                    num_experiments=num_experiments_per_val,
                    **{param_name: val, **experiment_kwargs}
                ),
                'status': 'Not Run',
                'results': None,
                'index': i,
            }
            for i, val in enumerate(param_vals)
        ]

        self.save_location = save_location
        if save_location is not None:
            self.load_history(save_location)
    
    def get_experiment_results(self, meta_experiment):
        return meta_experiment.results

    def _get_results(self):
        return [
            item['results'] for item in self.experiments
        ]
    
    def print_prehistory(self):
        pass
    
    def load_sub_experiment(self, experiment, path):
        history_path = path / 'training_history.json'
        history = json.load(history_path.open(mode='r'))
        
        results_path = path / 'results.json'
        if results_path.exists():
            results = json.load(results_path.open(mode='r'))
        else:
            results = [
                item['test_metrics'] for item in history
                if 'test_metrics' in item
            ][-1]
        
        config_path = path / 'config.json'
        if config_path.exists():
        
            config = json.load(config_path.open(mode='r'))
        
            # assuming that the experiment is self-play
            experiment.student.load_weights(str(path / 'agent_weights'))

        else:
            config = {'epochs_optimised': len(history)}

        experiment.epoch = config['epochs_optimised']
        experiment.training_history = history
        experiment.results = results
        
        return experiment
    
    def load_meta_exp(self, meta_exp_path):
        metadata = json.load((meta_exp_path / 'meta.json').open(mode='r'))
        results = json.load((meta_exp_path / 'results.json').open(mode='r'))
        
        if 'experiment_config' in metadata:
            play_params = metadata['experiment_config']['play_params']
            param_val = play_params[self.param_name]
        
        elif self.param_name in metadata:
            param_val = metadata[self.param_name]
        
        i, *_ = [i for i, item in enumerate(self.experiments)
                 if item[self.param_name] == param_val]
        
        self.experiments[i]['status'] = 'Complete'
        self.experiments[i]['results'] = results
        meta_exp = self.experiments[i]['experiment']
        meta_exp.results = results

        j = 0
        for sub_exp_path in meta_exp_path.glob('*'):
            try: 
                if not sub_exp_path.is_file():
                    sub_exp = meta_exp.experiments[j]['experiment']
                    meta_exp.experiments[j]['experiment'] = \
                        self.load_sub_experiment(sub_exp, sub_exp_path)
                    meta_exp.experiments[j]['status'] = 'Complete' 
                    meta_exp.experiments[j]['results'] = \
                        meta_exp.experiments[j]['experiment'].results
                    j += 1
            except:
                pass
            
    def load_history(self, history_location):
        exp_path = Path(history_location)
        for meta_exp_path in exp_path.glob('*'):
            results_path = meta_exp_path / 'results.json'
            history_path = meta_exp_path / 'training_history.json'
            if results_path.exists() or history_path.exists():
                self.load_meta_exp(meta_exp_path)
    
    def export_experiment(self, experiment):
        if self.save_location is not None:
            
            experiment_config = \
                experiment.experiments[0]['experiment'].get_config()
            
            i, *_ = [i for i, x in enumerate(self.experiments)
                     if x['experiment'] == experiment]
            meta_data = {
                'index': i,
                'experiment_config': experiment_config
            }
            meta_data_path = \
                Path(f'{experiment.export_location}/meta.json')
            with meta_data_path.open(mode='w') as f:
                json.dump(meta_data, f)
            
            results_path = \
                Path(f'{experiment.export_location}/results.json')
            with results_path.open(mode='w') as f:
                json.dump(experiment.results, f)

            print('Saved experiment data at:', 
                  experiment.export_location)
