#!/usr/bin/env python

import os

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

job_directory = "%s/.job" %os.getcwd()

table = "PreProcessingTable_Li_11_13_2018.json"

for t in range(1,45):

    job_file = os.path.join(job_directory,"%s.job" %t)

    with open(job_file) as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % t)
        fh.writelines("#SBATCH --output=.out/%s.out\n" % t)
        fh.writelines("#SBATCH --error=.out/%s.err\n" % t)
        fh.writelines("#SBATCH --time=01:00:00\n")
        fh.writelines("#SBATCH --mem=8000\n")
        fh.writelines("#SBATCH --cpus-per-task=4\n")
        fh.writelines("#SBATCH --partition=giocomo\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=alexg8@stanford.edu\n")
        fh.writelines("python3 bashPreProcess.py -t %s -f %s\n" % (t,table))

    os.system("sbatch %s" %job_file)
