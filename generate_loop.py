from audiocraft.models import musicgen
import torch
import numpy as np
import argparse
from sklearn.preprocessing import normalize
import IPython.display as ipd
from datetime import datetime
import taglib
import os


def main(
    music_directory,
    descriptions,
    model_name="medium",
    duration=30,
    min_loop_length=13,
    num_loops=5,
    comparison_frames=200,
):
    if not os.path.exists(music_directory):
        raise Exception(f"music_directory does not exist: {music_directory}")
    model = musicgen.MusicGen.get_pretrained(model_name, device="cuda")
    model.set_generation_params(duration=duration)
    min_loop_length_frames = min_loop_length * model.frame_rate

    attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, None)
    assert prompt_tokens is None
    tokens = model._generate_tokens(attributes, prompt_tokens, progress=True)

    emb = model.compression_model.decode_latent(tokens)
    emb = emb.detach().cpu()
    emb = emb.numpy()
    full_embs = []
    for batch_idx, emb_b in enumerate(emb):
        nemb_b = normalize(emb_b)

        rolled_nemb_b = []
        for offset in range(comparison_frames):
            off_nemb_b = np.zeros_like(nemb_b)
            if offset == 0:
                off_nemb_b[:] = nemb_b
            else:
                off_nemb_b[:, offset:] = nemb_b[:, :-offset]
            rolled_nemb_b.append(off_nemb_b)

        nemb_b = np.concatenate(rolled_nemb_b, axis=0)
        cov = np.dot(nemb_b.T, nemb_b)
        cov = np.triu(cov, k=min_loop_length_frames)
        first_frame, last_frame = np.unravel_index(cov.argmax(), cov.shape)
        print(first_frame, last_frame)

        full_emb = []
        first_loop = emb_b[:, :last_frame]
        full_emb.append(first_loop)
        for _ in range(num_loops - 2):
            full_emb.append(emb_b[:, first_frame:last_frame])
        if num_loops > 1:
            full_emb.append(emb_b[:, first_frame:])
        full_embs.append(np.hstack(full_emb))

    # process each song separately since they may have different lengths
    for batch_idx, full_emb in enumerate(full_embs):
        with torch.no_grad():
            out = model.compression_model.decoder(
                torch.tensor(full_emb).unsqueeze(0).cuda()
            )
            wav = model.compression_model.postprocess(out)
            wav = wav.detach().cpu().squeeze(0)
        aud = ipd.Audio(wav, rate=32000)
        save_path = os.path.join(
            music_directory,
            f"{datetime.now().strftime('%H%M%S%m%d%Y')}_{batch_idx}.wav",
        )
        with open(save_path, mode="bw") as f:
            f.write(aud.data)
        with taglib.File(save_path, save_on_exit=True) as song:
            song.tags["TITLE"] = [descriptions[batch_idx]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--music_directory",
        action="store",
        dest="music_directory",
        required=True,
        help="Path to the music directory",
    )
    parser.add_argument(
        "--descriptions",
        action="store",
        dest="descriptions",
        required=True,
        nargs="+",
        help="prompts used to generate songs",
    )
    parser.add_argument(
        "--model_name",
        action="store",
        dest="model_name",
        default="medium",
        help="name of the MusicGen model to use ",
    )
    parser.add_argument(
        "--duration",
        action="store",
        dest="duration",
        default=30,
        type=int,
        help="Length of the original generated (s)",
    )
    parser.add_argument(
        "--min_loop_length",
        action="store",
        dest="min_loop_length",
        default=13,
        type=int,
        help="Minimum length for loops (s)",
    )
    parser.add_argument(
        "--num_loops",
        action="store",
        dest="num_loops",
        default=5,
        type=int,
        help="Number of loops",
    )
    parser.add_argument(
        "--comparison_frames",
        action="store",
        dest="comparison_frames",
        default=200,
        type=int,
        help="Number of embedding frames to compare for loop selection, \
             the MusicGen models have 50 frames per second",
    )
    args = parser.parse_args()
    main(**vars(args))
