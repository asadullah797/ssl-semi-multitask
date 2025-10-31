# from transformers.configuration_utils import PretrainedConfig

# class Wav2Vec2MultiTaskConfig(PretrainedConfig):
#     model_type = "wav2vec2-multitask"

#     def __init__(
#         self,
#         num_emotions=14,
#         num_phonemes=33,
#         num_speakers=373,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.num_emotions = num_emotions
#         self.num_phonemes = num_phonemes
#         self.num_speakers = num_speakers

# from transformers import Wav2Vec2Config
from transformers import PretrainedConfig

class Wav2Vec2MultiTaskConfig(PretrainedConfig):
    model_type = "wav2vec2-multitask"

    def __init__(
        self,
        num_phonemes=33,
        num_emotions=14,
        num_speakers=373,
        add_adapter=False,  # keep base attributes compatible
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_phonemes = num_phonemes
        self.num_emotions = num_emotions
        self.num_speakers = num_speakers
        self.add_adapter = add_adapter