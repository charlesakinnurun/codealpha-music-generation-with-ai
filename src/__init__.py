"""Music Generation with AI - core package.

Contains the reusable building blocks:

- ``config``     project paths and hyper-parameter dataclasses
- ``preprocess`` MIDI -> token sequences -> training tensors
- ``model``      configurable PyTorch LSTM model
- ``train``      training / validation / checkpointing logic
- ``generate``   sampling + MIDI writing
- ``audio``      optional MIDI -> WAV conversion
"""