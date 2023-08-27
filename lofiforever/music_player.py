import sys
import time
from collections import deque
import os

import multiprocessing as mp
import simpleaudio as sa
import taglib


class DirectoryObserver:
    def __init__(
        self,
        directory: str,
        unplayed_queue: mp.Queue,
        stop_signal: mp.Value,
        refresh_time: int = 1,
        last_scan_time: int = None,
    ) -> None:
        self.unplayed_queue = unplayed_queue
        self.stop_signal = stop_signal
        self.refresh_time = refresh_time
        self.directory = directory
        if last_scan_time is None:
            self.last_scan_time = time.time()
        else:
            self.last_scan_time = last_scan_time

        self.load_directory(self.directory)

    def load_directory(self, directory: str = None):
        if directory is None:
            directory = self.directory
        for file in os.listdir(directory):
            file_ne, ext = os.path.splitext(file)
            if ext not in (".wav"):
                continue
            self.unplayed_queue.put(os.path.join(directory, file))

    def get_new_files(
        self,
        last_scan_time,
        directory=None,
    ):
        if directory is None:
            directory = self.directory
        current_time = time.time()

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath) and os.path.getctime(filepath) > last_scan_time:
                self.unplayed_queue.put(filepath)

        return current_time

    def run(self, directory=None):
        if directory is None:
            directory = self.directory
        while not self.stop_signal.value:
            self.last_scan_time = self.get_new_files(
                last_scan_time=self.last_scan_time,
                directory=directory,
            )
            time.sleep(self.refresh_time)


class MusicQueuePlayer:
    def __init__(
        self,
        unplayed_queue: mp.Queue,
        played_queue: mp.Queue,
        command_queue: mp.Queue,
        stop_signal=mp.Value,
        display_value=mp.Value,
    ) -> None:
        self.current_play = None
        self.unplayed_queue = unplayed_queue
        self.played_queue = played_queue
        self.command_queue = command_queue
        self.stop_signal = stop_signal
        self.display_value = display_value

    def start_song(self, song_path):
        if not os.path.exists(song_path):
            self.current_play = None
            return
        self.current_play = sa.WaveObject.from_wave_file(song_path).play()

        display_str = os.path.split(song_path)[-1]
        with taglib.File(song_path) as song:
            if "TITLE" in song.tags:
                display_str += f" : {song.tags['TITLE'][0]}"
        self.display_value.value = display_str
        self.played_queue.put(song_path)

    def do_command(self, command):
        if command == "skip":
            if self.current_play is not None:
                self.current_play.stop()
                self.current_play = None

        else:
            print(f"unknown command: {command}")

    def run(self):
        try:
            while not self.stop_signal.value:
                if self.command_queue.qsize() > 0:
                    command = self.command_queue.get()
                    self.do_command(command)
                if self.current_play is not None and self.current_play.is_playing():
                    time.sleep(1)
                else:
                    if self.unplayed_queue.qsize() > 0:
                        song_path = self.unplayed_queue.get()
                        self.start_song(song_path)

                    elif self.played_queue.qsize() > 0:
                        song_path = self.played_queue.get()
                        self.start_song(song_path)
                    else:
                        time.sleep(1)
        finally:
            if self.current_play is not None:
                self.current_play.stop()
                sa.stop_all()
                print("Shut down simpleaudio successful")
