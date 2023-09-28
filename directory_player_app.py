from lofiforever.music_player import MusicQueuePlayer, DirectoryObserver
import argparse
import multiprocessing as mp
import time
from ctypes import c_wchar_p
import curses


def main(stdscr, music_directory, show_instructions):
    # Initialize curses
    curses.curs_set(0)  # Hide the cursor
    stdscr.nodelay(1)  # Make getch() non-blocking

    # subprocess set up
    manager = mp.Manager()
    unplayed_queue = manager.Queue()
    played_queue = manager.Queue()
    command_queue = manager.Queue()
    stop_signal = manager.Value("i", 0)
    display_value = manager.Value(c_wchar_p, "None")
    player = MusicQueuePlayer(
        unplayed_queue=unplayed_queue,
        played_queue=played_queue,
        command_queue=command_queue,
        stop_signal=stop_signal,
        display_value=display_value,
    )
    dir_obs = DirectoryObserver(
        music_directory,
        unplayed_queue=unplayed_queue,
        played_queue=played_queue,
        stop_signal=stop_signal,
    )
    dir_obs.load_directory(load_as_played=True)
    player_proc = mp.Process(target=player.run)
    dir_proc = mp.Process(target=dir_obs.run)
    player_proc.start()
    dir_proc.start()

    # main loop
    try:
        while True:
            key = stdscr.getch()
            if key == ord("q"):
                break
            if key == ord("s"):
                command_queue.put("skip")
            # Clear the screen
            stdscr.clear()

            # Get height and width of the window
            h, w = stdscr.getmaxyx()

            center_text = display_value.value[: w - 2]
            center_x = max((0, w // 2 - len(center_text) // 2))
            center_y = h // 2
            try:
                stdscr.addstr(center_y, center_x, center_text)
            except curses.error:
                pass

            # Display instructions at the bottom
            if show_instructions:
                instructions = "Press 'q' to quit, 's' to skip song"
                try:
                    stdscr.addstr(h - 1, 0, instructions[: w - 2])
                except curses.error:
                    pass
            # Refresh the screen
            stdscr.refresh()
            time.sleep(1)
    finally:
        stop_signal.value = 1
        player_proc.join()
        dir_proc.join()
        print("processes joined")


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
        "--show_instructions",
        action=argparse.BooleanOptionalAction,
        dest="show_instructions",
        default=False,
        help="Path to the music directory",
    )
    args = parser.parse_args()
    curses.wrapper(main, args.music_directory, args.show_instructions)
