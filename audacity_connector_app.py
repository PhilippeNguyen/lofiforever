import argparse
from pipeclient import PipeClient
import shlex


import traceback
import readline
from app_commands import command_dict


class AudMainApp:
    def __init__(self, timeout=10):
        self.audacity_pipe = PipeClient()
        self.timeout = timeout

        self.models = {}

        self.commands = {
            command_class.name: command_class(self)
            for command_class in command_dict.values()
        }
        self.parser = argparse.ArgumentParser(
            description="A program with multiple commands"
        )
        self.build_subparsers()

        self.main_loop()

    def build_subparsers(self):
        self.subparsers = self.parser.add_subparsers(
            dest="command", help="Available commands"
        )
        for command_name, command_class in self.commands.items():
            command_class.add_subparser(self.subparsers)

    def main_loop(self):
        print("Welcome to the command-line program!")
        while True:
            command = input("Enter a command (or 'exit' to quit): ")
            readline.add_history(command)
            if command == "exit":
                print("Exiting program.")
                break
            try:
                parsed_command = self.parser.parse_args(shlex.split(command))
            except (Exception, SystemExit) as e:
                print(e)
                continue

            if parsed_command.command in self.commands:
                try:
                    self.commands[parsed_command.command](**vars(parsed_command))
                except Exception as e:
                    tb = traceback.extract_tb(e.__traceback__)[-1]
                    lineno, filename, line = tb.lineno, tb.filename, tb.line
                    print(f"Error in file {filename} at line {lineno}: {line}")
                    print("Exception message:", e)
            else:
                print(f"Unknown command: {parsed_command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument(
    #     "--music_directory",
    #     action="store",
    #     dest="music_directory",
    #     required=True,
    #     help="Path to the music directory",
    # )
    # parser.add_argument(
    #     "--show_instructions",
    #     action=argparse.BooleanOptionalAction,
    #     dest="show_instructions",
    #     default=False,
    #     help="Path to the music directory",
    # )
    args = parser.parse_args()
    AudMainApp(**vars(args))
