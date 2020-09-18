from .play_game import maybe_mutate_message
from .loss import get_correct_teacher_msg


class SyntheticTeacher:
    
    def __init__(self, channel_size, num_classes, targets):
        self.channel_size = channel_size
        self.num_classes = num_classes
        self.targets = targets
    
    def __call__(self, inputs, state=None, training=False):
        inp, prev_utt, other_utt = inputs
        
        if state is not None:
            history = state
            history.append({
                'message_from_teacher': prev_utt,
            })
        else:
            history = []
        
        if len(history) < self.num_classes:
            utt, _ = maybe_mutate_message(prev_utt,
                                          self.channel_size,
                                          history,
                                          p_mutate=1)
        else:
            utt = get_correct_teacher_msg(history + [history[0]], 
                                          self.targets)
        
        state = history
        
        # utterances are logits not probabilities
        utt = 1000*utt
        
        return utt, None, state 
