import seewav
import argparse
import os
import tempfile
from pathlib import Path
import taglib
import numpy as np
import colorsys


def yiq_background(foreground_rgb):
    R, G, B = foreground_rgb
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return (0, 0, 0) if Y > 0.5 else (1, 1, 1)


def luminance_background(foreground_rgb):
    R, G, B = foreground_rgb
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return (0, 0, 0) if L > 0.5 else (1, 1, 1)


def complementary_background(foreground_rgb):
    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(
        foreground_rgb[0], foreground_rgb[1], foreground_rgb[2]
    )

    # Adjust the hue by 180 degrees
    h = (h + 0.5) % 1.0

    # Convert HSL back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    return r, g, b


background_funcs = [
    yiq_background,
    luminance_background,
    complementary_background,
    complementary_background,
]


def random_pick_background(foreground_rgb):
    background_func = background_funcs[np.random.choice(3)]
    return background_func(foreground_rgb=foreground_rgb)


def main(music_directory, video_directory, fg_color=None, bg_color=None):
    os.makedirs(video_directory, exist_ok=True)
    for file in os.listdir(music_directory):
        song_path = os.path.join(music_directory, file)
        try:
            file_ne, ext = os.path.splitext(file)
            out = os.path.join(video_directory, file_ne + ".mp4")

            if fg_color is None:
                fg_color = np.random.random(3).tolist()
            if bg_color is None:
                bg_color = random_pick_background(foreground_rgb=fg_color)
            with tempfile.TemporaryDirectory() as tmp:
                seewav.visualize(
                    Path(song_path),
                    Path(tmp),
                    Path(out),
                    fg_color=fg_color,
                    bg_color=bg_color,
                )
            title_str = None
            with taglib.File(song_path) as song:
                if "TITLE" in song.tags:
                    title_str = f"{song.tags['TITLE'][0]}"
            if title_str is not None:
                with taglib.File(out, save_on_exit=True) as video_file:
                    video_file.tags["TITLE"] = [title_str]
        except Exception as e:
            print(e)


def parse_color(colorstr):
    """
    Given a comma separated rgb(a) colors, returns a 4-tuple of float.
    """
    try:
        r, g, b = [float(i) for i in colorstr.split(",")]
        return r, g, b
    except ValueError:
        print(
            "Format for color is 3 floats separated by commas 0.xx,0.xx,0.xx, rgb order"
        )
        raise


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
        "--video_directory",
        action="store",
        dest="video_directory",
        required=True,
        help="Path to the music directory",
    )
    parser.add_argument(
        "--bg_color",
        default=None,
        type=parse_color,
        dest="bg_color",
        help="Color of the background as `r,g,b` in [0, 1].",
    )
    parser.add_argument(
        "--fg_color",
        default=None,
        type=parse_color,
        dest="fg_color",
        help="Color of the bars as `r,g,b` in [0, 1].",
    )
    args = parser.parse_args()
    main(**vars(args))
