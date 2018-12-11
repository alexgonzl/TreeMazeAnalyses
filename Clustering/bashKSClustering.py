#!/usr/bin/env python

import os

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" %os.getcwd()
mkdir_p(job_directory)

table = "Clustering_Ne_12_7_2018.json"
nJobs = 62

for t in range(1,nJobs+1):

    job_file = os.path.join(job_directory,"t%s.job" %t)

    with open(job_file,"w+") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("ml matlab/R2018a\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % t)
        fh.writelines("#SBATCH --error=.out/%s.err\n" % t)
        fh.writelines("#SBATCH --time=08:00:00\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fun = "try matlabSherlockBashSession(%s,'%s'); catch; end; quit" % (t,table)
        fh.writelines('matlab -r "%s" \n' % (fun))

    os.system("sbatch --partition=giocomo,owners --mem=16000 --cpus-per-task=8 --mail-user=alexg8@stanford.edu %s" %job_file)
