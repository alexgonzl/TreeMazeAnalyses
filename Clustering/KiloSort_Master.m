function KiloSort_Master(datFile,datDir)
% default options are in parenthesis after the comment
kilopath = '../Lib/KiloSort/';
npy_mat  = '../Lib/npy-matlab/';
addpath(genpath(kilopath)) % path to kilosort folder
addpath(genpath(npy_mat)) % path to npy-matlab scripts
fn = datFile;
sp = datDir;

%pathToYourConfigFile = './'; % take from Github folder and put it somewhere else (together with the master_file)
%run(fullfile(pathToYourConfigFile, 'KiloSort_Config.m')) % change into function that can take input files.
ops = KiloSort_Config(fn,sp);

if (~exist(fullfile(sp,'rez.mat')) || ops.Overwrite)
    tic; % start timer
    %
    if ops.GPU     
        gpuDevice(1); % initialize GPU (will erase any existing GPU arrays)
    end

    if strcmp(ops.datatype , 'openEphys')
       ops = convertOpenEphysToRawBInary(ops);  % convert data, only for OpenEphys
    end
    %
    [rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
    rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
    rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)

    % AutoMerge. rez2Phy will use for clusters the new 5th column of st3 if you run this)
    rez = merge_posthoc2(rez);

    % save matlab results file
    save(fullfile(ops.root,  'rez.mat'), 'rez', '-v7.3');

    % save python results file for Phy
    rezToPhy(rez, ops.root);

    % remove temporary file
    delete(ops.fproc);
else
    disp('File already exists and overwrite=0')
end
