from pathlib import Path
import sys
import time

class recording_session(object):
    def __init__(self, session_name):
        self.sessionID = Path(session_name)
        self.get_csc_files()
        self.get_vt_files()
        self.get_ev_files()

    def get_csc_files(self):
        self.csc_files =[]
        for i in self.sessionID.glob('*.ncs'):
            if i.stat().st_size>16384:
                self.csc_files.append((i.name, i.stem, str(i.absolute()), time.ctime(i.stat().st_mtime)))

    def get_vt_files(self):
        self.vt_files =[]
        for i in fp.glob('*.nvt'):
            if i.stat().st_size>16384:
                self.vt_files.append((i.name, i.stem, str(i.absolute()),  time.ctime(i.stat().st_mtime)))

    def get_ev_files(self):
        self.ev_files =[]
        for i in fp.glob('*.nev'):
            if i.stat().st_size>16384:
                self.ev_files.append((i.name, i.stem, str(i.absolute()),  time.ctime(i.stat().st_mtime)))

    def get_tetrode_files(self):
        if len(self.csc_files)>0:
            
    def create_file_table(self):
