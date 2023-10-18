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
import numpy as np
from sklearn.preprocessing import normalize as sk_normalize
import json


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

    def _get_current_selection(self):
        init_labels = self.get_label_info()
        # selected_tracks = self._get_current_selected_tracks()
        # if len(selected_tracks) == 0:
        #     raise Exception("No tracks selected")
        # selected_track = selected_tracks[0]  # only take one track
        # self.main_app.audacity_pipe.do_command_and_wait(f"NewLabelTrack:")
        # self.main_app.audacity_pipe.do_command_and_wait(
        #     f"SelectTracks: Track={selected_track}"
        # )
        self.main_app.audacity_pipe.do_command_and_wait(f"GetSelection:")
        new_labels = self.get_label_info()
        new_label = new_labels.difference(init_labels).pop()
        new_label_text = new_label[3]
        start_txt, end_txt = new_label_text.split(",")
        start = float(start_txt.split(": ")[1])
        end = float(end_txt.split(": ")[1])
        self.main_app.audacity_pipe.do_command_and_wait(f"LastTrack:")
        self.main_app.audacity_pipe.do_command_and_wait(f"TrackClose:")
        return start, end

    def _get_current_selected_tracks(self):
        info_reply = self.main_app.audacity_pipe.do_command_and_wait(
            f"GetInfo: Type=Tracks", print_response=False
        )
        track_list = json.loads(info_reply.split("\nBatchCommand")[0])
        results = []
        for track_id, track_data in enumerate(track_list):
            if track_data["selected"] == 1:
                results.append(track_id)
        return results

    def _load_current_selected_audio(self):
        _, reg_file_path = tempfile.mkstemp(suffix=".wav")
        self.main_app.audacity_pipe.do_command_and_wait(
            f"Export2: Filename={reg_file_path} NumChannels=1.0"
        )
        # self.main_app.audacity_pipe.do_command_and_wait(f"SelSave:")
        reg_waveform, reg_sr = torchaudio.load(reg_file_path)
        os.remove(reg_file_path)
        return reg_waveform, reg_sr

    def get_label_info(self):
        info_reply = self.main_app.audacity_pipe.do_command_and_wait(
            f"GetInfo: Type=Labels", print_response=False
        )
        label_info = json.loads(info_reply.split("\nBatchCommand")[0])
        return self.parse_label_info(label_info)

    def parse_label_info(self, label_info_obj):
        num_label_tracks = len(label_info_obj)
        total_num_labels = 0
        label_set = set()
        for label_track_id, labels in label_info_obj:
            total_num_labels += len(labels)
            for label in labels:
                start, end, text = label
                label_set.add((label_track_id, start, end, text))

        return label_set


class Similarity(Commands):
    name = "similarity"

    @staticmethod
    def add_subparser(subparsers):
        parser = subparsers.add_parser(
            Similarity.name,
            help="Compares selected region with another to check for similar  \
                regions. Will ask for additional input after selecting this \
                command.",
        )
        parser.add_argument(
            "--self",
            dest="self_similarity",
            action="store_true",
            default=False,
            help="If True comare selected region with itself, otherwise compare \
                two different selections",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.0,
            help="Only return regions which pass this threshold for similarity \
                not used if --best used.",
        )
        parser.add_argument(
            "--comparison_seconds",
            type=float,
            default=2,
            help="Number of seconds used for comparison",
        )
        parser.add_argument(
            "--min_loop_length_seconds",
            type=float,
            default=None,
            help="Number of seconds elapsed after the source to compare with",
        )
        parser.add_argument(
            "--return_limit",
            type=int,
            default=1,
            help="Max number of regions/labels to create",
        )

        parser.add_argument(
            "--nms_seconds",
            type=int,
            default=1,
            help="Do not create labels that start/end within this many seconds of \
                a label with higher similarity",
        )
        return parser

    def __call__(
        self,
        self_similarity=True,
        threshold=0.0,
        comparison_seconds=2,
        min_loop_length_seconds=None,
        return_limit=10,
        nms_seconds=1,
        **kwargs,
    ):
        if "musicgen" not in self.main_app.models:
            raise Exception("MusicGen model not loaded yet")
        model = self.main_app.models["musicgen"]
        if self_similarity:
            if min_loop_length_seconds is None:
                min_loop_length_seconds = 2
            self._self_similarity(
                model=model,
                threshold=threshold,
                comparison_seconds=comparison_seconds,
                min_loop_length_seconds=min_loop_length_seconds,
                return_limit=return_limit,
                nms_seconds=nms_seconds,
            )
        else:
            if min_loop_length_seconds is None:
                min_loop_length_seconds = 0
            self._two_similarity(
                model=model,
                threshold=threshold,
                comparison_seconds=comparison_seconds,
                min_loop_length_seconds=min_loop_length_seconds,
                return_limit=return_limit,
                nms_seconds=nms_seconds,
            )

    def _self_similarity(
        self,
        model: musicgen.MusicGen,
        threshold: float = 0.0,
        comparison_seconds=2,
        min_loop_length_seconds=2,
        return_limit=10,
        nms_seconds=1,
    ):
        self.main_app.audacity_pipe.do_command_and_wait(f"SelSave:")
        select_start, select_end = self._get_current_selection()
        reg_waveform, reg_sr = self._load_current_selected_audio()
        cm = model.compression_model
        comparison_frames = int(comparison_seconds * cm.frame_rate)
        min_loop_length_frames = int(min_loop_length_seconds * cm.frame_rate)
        nms_frames = int(nms_seconds * cm.frame_rate)

        emb = self.get_embedding(compression_model=cm, waveform=reg_waveform)
        rolled_emb = self.build_delay_array(
            audio_embedding=emb, comparison_frames=comparison_frames
        )
        cov = np.dot(rolled_emb.T, rolled_emb)
        cov = np.triu(cov, k=min_loop_length_frames)

        top_values = self.non_max_suppression(
            cov, k=return_limit, radius=nms_frames, threshold=threshold
        )

        # Need to add new track, and sort labels by start loc because audacity
        # ids labels by track, then by start location
        self.main_app.audacity_pipe.do_command_and_wait(f"NewLabelTrack:")
        top_values.sort(key=lambda x: x[1][0])
        for similarity, loc in top_values:
            row_idx, col_idx = loc

            first_seconds, last_seconds = (
                row_idx / cm.frame_rate,
                col_idx / cm.frame_rate,
            )
            self.create_similarity_label(
                start_time=first_seconds + select_start,
                end_time=last_seconds + select_start,
                label=similarity,
            )

    def _two_similarity(
        self,
        model: musicgen.MusicGen,
        threshold: float = 0.0,
        comparison_seconds=2,
        min_loop_length_seconds=0,
        return_limit=10,
        nms_seconds=1,
    ):
        input("Make a first selection then press Enter:")

        first_start, first_end = self._get_current_selection()
        first_waveform, _ = self._load_current_selected_audio()

        cm = model.compression_model
        comparison_frames = int(comparison_seconds * cm.frame_rate)
        min_loop_length_frames = int(min_loop_length_seconds * cm.frame_rate)
        nms_frames = int(nms_seconds * cm.frame_rate)

        first_emb = self.get_embedding(compression_model=cm, waveform=first_waveform)
        rolled_first_emb = self.build_delay_array(
            audio_embedding=first_emb, comparison_frames=comparison_frames
        )
        input("Make a second selection then press Enter:")

        second_start, second_end = self._get_current_selection()
        second_waveform, _ = self._load_current_selected_audio()

        second_emb = self.get_embedding(compression_model=cm, waveform=second_waveform)
        rolled_second_emb = self.build_delay_array(
            audio_embedding=second_emb, comparison_frames=comparison_frames
        )
        cov = np.dot(rolled_first_emb.T, rolled_second_emb)
        cov = np.triu(cov, k=min_loop_length_frames)

        top_values = self.non_max_suppression(
            cov, k=return_limit, radius=nms_frames, threshold=threshold
        )
        self.main_app.audacity_pipe.do_command_and_wait(f"NewLabelTrack:")
        top_values.sort(key=lambda x: x[1][0])
        for similarity, loc in top_values:
            row_idx, col_idx = loc

            first_seconds, last_seconds = (
                row_idx / cm.frame_rate,
                col_idx / cm.frame_rate,
            )
            self.create_similarity_label(
                start_time=first_seconds + first_start,
                end_time=last_seconds + second_start,
                label=similarity,
            )

    def get_embedding(self, compression_model, waveform, normalize=True):
        with torch.no_grad():
            reg_codes, reg_scale = compression_model.preprocess(
                torch.unsqueeze(waveform.cuda(), 0)
            )
            reg_emb = compression_model.encoder(reg_codes).detach().cpu()

        # Explicitly delete tensors
        del reg_codes
        del reg_scale

        # Empty the GPU cache
        torch.cuda.empty_cache()

        emb_np = reg_emb.numpy()[0]
        if normalize:
            emb_np = sk_normalize(emb_np)
        return emb_np

    def build_delay_array(self, audio_embedding, comparison_frames):
        rolled_emb = []
        for offset in range(comparison_frames):
            off_nemb_b = np.zeros_like(audio_embedding)
            if offset == 0:
                off_nemb_b[:] = audio_embedding
            else:
                off_nemb_b[:, offset:] = audio_embedding[:, :-offset]
            rolled_emb.append(off_nemb_b)

        return np.concatenate(rolled_emb, axis=0)

    def non_max_suppression(self, matrix, k, radius=1, threshold=0):
        suppressed_matrix = matrix.copy()
        top_values = []

        for _ in range(k):
            # Find the location of the maximum value in the matrix
            max_val = suppressed_matrix.max()
            if max_val < threshold:  # No more valid values
                break
            max_loc = np.unravel_index(
                suppressed_matrix.argmax(), suppressed_matrix.shape
            )

            # Add the max value and its location to the top values list
            top_values.append((max_val, max_loc))

            # Suppress neighboring values
            x_min = max(0, max_loc[0] - radius)
            x_max = min(matrix.shape[0], max_loc[0] + radius + 1)
            y_min = max(0, max_loc[1] - radius)
            y_max = min(matrix.shape[1], max_loc[1] + radius + 1)

            suppressed_matrix[x_min:x_max, y_min:y_max] = -np.inf

        return top_values

    def create_similarity_label(self, start_time, end_time, label):
        info_reply = self.main_app.audacity_pipe.do_command_and_wait(
            f"GetInfo: Type=Labels", print_response=False
        )
        label_info = json.loads(info_reply.split("\nBatchCommand")[0])
        num_labels = len(self.parse_label_info(label_info))
        # self.main_app.audacity_pipe.do_command_and_wait(f"SelRestore:")
        self.main_app.audacity_pipe.do_command_and_wait(
            f"Select: \
                Start={start_time} End={end_time} \
                    RelativeTo=ProjectStart"
        )
        self.main_app.audacity_pipe.do_command_and_wait(f"AddLabel:")
        self.main_app.audacity_pipe.do_command_and_wait(
            f"SetLabel: Label={num_labels} Text={label:.4f}"
        )


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
        # _, file_path = tempfile.mkstemp(suffix=".wav")
        # self.main_app.audacity_pipe.do_command_and_wait(
        #     f"Export2: Filename={file_path} NumChannels=1.0"
        # )
        self.main_app.audacity_pipe.do_command_and_wait(f"SelSave:")
        self.main_app.audacity_pipe.do_command_and_wait(f"SelRestore:")
        # init_waveform, init_sr = torchaudio.load(file_path)
        init_waveform, init_sr = self._load_current_selected_audio()
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
        # os.remove(file_path)


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
