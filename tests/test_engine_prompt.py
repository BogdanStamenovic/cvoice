"""The cached clone prompt must be built under inference mode.

Regression for ba73414: built bare, create_voice_clone_prompt traced gradients
through the DAC encoder and a 20 s reference took cvoiced to 7.5 GiB (OOM).
Uses a stand-in model so it runs without a GPU or the weights.
"""

import torch

from cvoice.engine import Engine


class FakeModel:
    def __init__(self):
        self.prompt_built_in_inference = None
        self.prompts_built = 0

    def create_voice_clone_prompt(self, ref_audio, ref_text=None):
        self.prompt_built_in_inference = torch.is_inference_mode_enabled()
        self.prompts_built += 1
        return object()

    def generate(self, text, voice_clone_prompt=None, **kwargs):
        self.kwargs = kwargs
        return [torch.zeros(10)]


def engine_with(model):
    e = Engine()
    e._model = model
    e.load = lambda: model
    return e


def test_prompt_built_under_inference_mode(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x")
    m = FakeModel()
    engine_with(m).speak("Zdravo.", str(ref), "ref text")
    assert m.prompt_built_in_inference is True


def test_prompt_cached_until_reference_changes(tmp_path):
    import os

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x")
    m = FakeModel()
    e = engine_with(m)
    e.speak("a", str(ref), "t")
    e.speak("b", str(ref), "t")
    assert m.prompts_built == 1
    os.utime(ref, (1, 1))  # re-enrolment rewrites the file
    e.speak("c", str(ref), "t")
    assert m.prompts_built == 2


def test_steps_passed_only_when_given(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x")
    m = FakeModel()
    e = engine_with(m)
    e.speak("a", str(ref), "t")
    assert m.kwargs == {}
    e.speak("a", str(ref), "t", steps=8)
    assert m.kwargs == {"num_step": 8}
