# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# torchaudio 2.9 removed torchaudio.info() and moved torchaudio.load() to torchcodec (which needs FFmpeg).
# openwakeword.data and torch_audiomentations call both, so on those versions they are replaced with
# soundfile-based equivalents. Importing this module applies the patch; older torchaudio is left alone.

from types import SimpleNamespace

import soundfile
import torch
import torchaudio


def _info(filepath, *args, **kwargs):
    info = soundfile.info(str(filepath))
    return SimpleNamespace(sample_rate=info.samplerate, num_frames=info.frames, num_channels=info.channels)


def _load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, *args, **kwargs):
    data, sample_rate = soundfile.read(str(filepath), start=frame_offset, frames=num_frames,
                                       dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data)
    return (waveform.T.contiguous() if channels_first else waveform), sample_rate


_version = tuple(int(i) for i in torchaudio.__version__.split("+")[0].split(".")[:2])
if _version >= (2, 9) or not hasattr(torchaudio, "info"):
    torchaudio.info = _info
    torchaudio.load = _load
