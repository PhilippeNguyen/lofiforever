from abc import ABC
from audiocraft.models import musicgen
import tempfile
import time
import typing as tp
import IPython.display as ipd
import os
import torchaudio
import torch
import demucs.api


class Commands(ABC):
    def __init__(self, main_app):
        self.main_app = main_app

    @property
    def name(self):
        pass

    @staticmethod
    def add_subparser():
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class SimilarityMatch(Commands):
    name = "similarity_match"

    @staticmethod
    def add_subparser(subparsers):
        parser = subparsers.add_parser(
            Separate.name,
            help="Seperate selected audio into components",
        )
        parser.add_argument(
            "--segment",
            type=int,
            default=None,
            help="Length of segments to separate individually (smaller segments save memory)",
        )
        return parser

    def __call__(
        self,
    ):
        pass


# class SelfSimilarityMatch(Commands):
#     pass


class Separate(Commands):
    name = "separate"

    @staticmethod
    def add_subparser(subparsers):
        parser = subparsers.add_parser(
            Separate.name,
            help="Seperate selected audio into components",
        )
        parser.add_argument(
            "--segment",
            type=int,
            default=None,
            help="Length of segments to separate individually (smaller segments save memory)",
        )
        return parser

    def __call__(self, segment=None, **kwargs):
        if "demucs" not in self.main_app.models:
            raise Exception("demucs model not loaded yet")
        model = self.main_app.models["demucs"]
        self._separate(model, segment=segment)

    def _separate(self, model: demucs.api.Separator, segment=None, **kwargs):
        if segment is not None:
            prev_segment = model._segment
            model.update_parameter(segment=segment)
        _, file_path = tempfile.mkstemp(suffix=".wav")
        self.main_app.audacity_pipe.do_command_and_wait(
            f"Export2: Filename={file_path} NumChannels=1.0"
        )
        self.main_app.audacity_pipe.do_command_and_wait(f"SelSave:")
        self.main_app.audacity_pipe.do_command_and_wait(f"SelRestore:")

        origin, separated = model.separate_audio_file(file_path)
        for stem, source in separated.items():
            _, genfile_path = tempfile.mkstemp(suffix=".wav")
            demucs.api.save_audio(source, genfile_path, samplerate=model.samplerate)
            self.main_app.audacity_pipe.do_command_and_wait(
                f"Import2: Filename={genfile_path}"
            )
            self.main_app.audacity_pipe.do_command_and_wait(f"SelRestore:")
            self.main_app.audacity_pipe.do_command_and_wait(f"Align_StartToSelStart:")
            os.remove(genfile_path)
        os.remove(file_path)
        if segment is not None:
            model.update_parameter(segment=prev_segment)


class Generate(Commands):
    name = "generate"

    @staticmethod
    def add_subparser(subparsers):
        parser = subparsers.add_parser(
            Generate.name,
            help="Generate audio for a given prompt",
        )
        parser.add_argument(
            "--duration",
            type=int,
            default=30,
            help="How many extra seconds to generate",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            default=None,
            help="Description of the music to output",
        )
        return parser

    def __call__(
        self, prompt: tp.Optional[str] = None, duration: tp.Optional[int] = 30, **kwargs
    ):
        if "musicgen" not in self.main_app.models:
            raise Exception("MusicGen model not loaded yet")
        model = self.main_app.models["musicgen"]
        self._generate(model, prompt=prompt, duration=duration)

    def _generate(
        self,
        model: musicgen.MusicGen,
        prompt: tp.Optional[str] = None,
        duration: tp.Optional[int] = 30,
    ):
        self.main_app.audacity_pipe.do_command_and_wait(f"SelSave:")
        self.main_app.audacity_pipe.do_command_and_wait(f"SelRestore:")
        model.set_generation_params(duration=duration)

        if prompt is None:
            prompt = [None]
        else:
            prompt = [prompt]

        wav = model.generate(
            descriptions=prompt,
            progress=True,
        )
        wav = wav.detach().cpu()[0]
        aud = ipd.Audio(wav, rate=model.sample_rate)
        _, genfile_path = tempfile.mkstemp(suffix=".wav")
        with open(genfile_path, mode="bw") as f:
            f.write(aud.data)
        self.main_app.audacity_pipe.do_command_and_wait(
            f"Import2: Filename={genfile_path}"
        )
        self.main_app.audacity_pipe.do_command_and_wait(f"SelRestore:")
        self.main_app.audacity_pipe.do_command_and_wait(f"Align_StartToSelStart:")
        os.remove(genfile_path)


class GenerateContinuation(Commands):
    name = "generate_continuation"

    @staticmethod
    def add_subparser(subparsers):
        parser = subparsers.add_parser(
            GenerateContinuation.name,
            help=" Generate continuation to a selected portion of audio",
        )
        parser.add_argument(
            "--duration",
            type=int,
            default=30,
            help="How many extra seconds to generate",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            default=None,
            help="Description of the music to output",
        )
        return parser

    def __call__(
        self, prompt: tp.Optional[str] = None, duration: tp.Optional[int] = 30, **kwargs
    ):
        if "musicgen" not in self.main_app.models:
            raise Exception("MusicGen model not loaded yet")
        model = self.main_app.models["musicgen"]
        self._generate_continuation(model, prompt=prompt, duration=duration)

    def _generate_continuation(
        self,
        model: musicgen.MusicGen,
        prompt: tp.Optional[str] = None,
        duration: tp.Optional[int] = 30,
    ):
        model.set_generation_params(duration=duration)
        _, file_path = tempfile.mkstemp(suffix=".wav")
        self.main_app.audacity_pipe.do_command_and_wait(
            f"Export2: Filename={file_path} NumChannels=1.0"
        )
        self.main_app.audacity_pipe.do_command_and_wait(f"SelSave:")
        self.main_app.audacity_pipe.do_command_and_wait(f"SelRestore:")
        init_waveform, init_sr = torchaudio.load(file_path)

        if prompt is None:
            prompt = [None]
        else:
            prompt = [prompt]

        wav = model.generate_continuation(
            prompt=init_waveform,
            descriptions=prompt,
            prompt_sample_rate=init_sr,
            progress=True,
        )
        wav = wav.detach().cpu()[0]

        gain = torch.mean(torch.abs(wav[:, : init_waveform.shape[1]])) / torch.mean(
            torch.abs(init_waveform)
        )
        wav = wav / gain
        aud = ipd.Audio(wav, rate=model.sample_rate)
        _, genfile_path = tempfile.mkstemp(suffix=".wav")
        with open(genfile_path, mode="bw") as f:
            f.write(aud.data)
        self.main_app.audacity_pipe.do_command_and_wait(
            f"Import2: Filename={genfile_path}"
        )
        self.main_app.audacity_pipe.do_command_and_wait(f"SelRestore:")
        self.main_app.audacity_pipe.do_command_and_wait(f"Align_StartToSelStart:")
        os.remove(genfile_path)
        os.remove(file_path)


# class LoadMusicGen(Commands):
#     name = "load_musicgen"

#     @staticmethod
#     def add_subparser(subparsers):
#         parser = subparsers.add_parser(LoadMusicGen.name, help="Load a MusicGen model.")
#         parser.add_argument(
#             "model_name",
#             nargs="?",
#             default=None,
#             help="Name of the MusicGen model to load. ('small'/'medium'/'large')",
#         )
#         parser.add_argument(
#             "--model_name",
#             dest="model_name",
#             type=str,
#             help="Name of the  MusicGen model to load (using keyword) \
#             ('small'/'medium'/'large').",
#         )
#         return parser

#     def __call__(self, model_name, **kwargs):
#         if model_name is None:
#             raise ValueError("model_name must be given")
#         self.main_app.models["musicgen"] = musicgen.MusicGen.get_pretrained(
#             model_name, device="cuda"
#         )


class Load(Commands):
    name = "load"

    @staticmethod
    def add_subparser(base_parser):
        parser = base_parser.add_parser(Load.name, help="Load a model.")

        #
        subparsers = parser.add_subparsers(dest="model_type", help="model type to use")
        musicgen_parser = subparsers.add_parser(
            "musicgen", help="Load a MusicGen model."
        )
        musicgen_parser.add_argument(
            "model_name",
            nargs="?",
            default=None,
            help="Name of the MusicGen model to load. ('small'/'medium'/'large')",
        )
        musicgen_parser.add_argument(
            "--model_name",
            dest="model_name",
            type=str,
            help="Name of the  MusicGen model to load (using keyword) \
            ('small'/'medium'/'large').",
        )

        demucs_parser = subparsers.add_parser("demucs", help="Load a demucs model")
        demucs_parser.add_argument(
            "model_name",
            nargs="?",
            default=None,
            help="Name of the demucs model to load. ('small'/'medium'/'large')",
        )
        demucs_parser.add_argument(
            "--model_name",
            dest="model_name",
            type=str,
            help="Name of the  MusicGen model to load (using keyword) \
            ('small'/'medium'/'large').",
        )
        demucs_parser.add_argument(
            "--segment",
            dest="segment",
            type=int,
            default=None,
            help="Length of segments to separate individually (smaller segments save memory)",
        )
        return parser

    def __call__(self, model_type, **kwargs):
        if model_type is None:
            raise ValueError("model_type must be given")
        elif model_type == "musicgen":
            self.load_musicgen(**kwargs)
        elif model_type == "demucs":
            self.load_demucs(**kwargs)
        else:
            raise Exception(f"Unknown model type {model_type}, see load --help")

    def load_musicgen(self, model_name, **kwargs):
        self.main_app.models["musicgen"] = musicgen.MusicGen.get_pretrained(
            model_name, device="cuda"
        )
        print(f"music_gen {model_name} loaded")

    def load_demucs(self, model_name, segment, **kwargs):
        demucs_args = {}
        if model_name is not None:
            demucs_args["model"] = model_name
        if segment is not None:
            demucs_args["segment"] = segment
        self.main_app.models["demucs"] = demucs.api.Separator(**demucs_args)
        print(f"demucs {self.main_app.models['demucs']._name} loaded")


def collect_local_classes(d):
    skip_classes = {"ABC", "Commands"}
    return {
        name: cls
        for name, cls in d.items()
        if isinstance(cls, type)
        and not name.startswith("__")
        and name not in skip_classes
    }


command_dict = collect_local_classes(globals())
