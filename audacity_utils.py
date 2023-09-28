import os
import sys


class AudPipeManager:
    def __init__(self):
        print(" Starting audacity pipe manager")
        if sys.platform == "win32":
            self.TONAME = "\\\\.\\pipe\\ToSrvPipe"
            self.FROMNAME = "\\\\.\\pipe\\FromSrvPipe"
            self.EOL = "\r\n\0"
        else:
            self.TONAME = "/tmp/audacity_script_pipe.to." + str(os.getuid())
            self.FROMNAME = "/tmp/audacity_script_pipe.from." + str(os.getuid())
            self.EOL = "\n"
        print('Write to  "' + self.TONAME + '"')
        if not os.path.exists(self.TONAME):
            print(
                " ..does not exist.  Ensure Audacity is running with mod-script-pipe."
            )
            sys.exit()

        print('Read from "' + self.FROMNAME + '"')
        if not os.path.exists(self.FROMNAME):
            print(
                " ..does not exist.  Ensure Audacity is running with mod-script-pipe."
            )
            sys.exit()
        print("-- Both pipes exist.  Good.")

        self.TOFILE = open(self.TONAME, "w")
        print("-- File to write to has been opened")
        self.FROMFILE = open(self.FROMNAME, "rt")
        print("-- File to read from has now been opened too\r\n")

    def send_command(self, command):
        """Send a single command."""
        print("Send: >>> \n" + command)
        self.TOFILE.write(command + self.EOL)
        self.TOFILE.flush()

    def get_response(self):
        """Return the command response."""
        result = ""
        line = ""
        while True:
            result += line
            line = self.FROMFILE.readline()
            if line == "\n" and len(result) > 0:
                break
        return result

    def do_command(self, command, print_response=True):
        """Send one command, and return the response."""
        self.send_command(command)
        response = self.get_response()
        if print_response:
            print("Rcvd: <<< \n" + response)
        return response
