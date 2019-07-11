function KiloSort_Master(fn,hfn,sp,ftype)
% default options are in parenthesis after the comment

kilopath = '../Lib/KiloSort/';
npy_mat  = '../Lib/npy-matlab/';
addpath(genpath(kilopath)) % path to kilosort folder
addpath(genpath(npy_mat)) % path to npy-matlab scripts

%pathToYourConfigFile = './'; % take from Github folder and put it somewhere else (together with the master_file)
%run(fullfile(pathToYourConfigFile, 'KiloSort_Config.m')) % change into function that can take input files.
ops = KiloSort_Config(fn,hfn,sp,ftype);

disp('')
disp(strcat('Processing File', fn))

tic; % start timer
%
if ops.GPU
    gpuDevice(1); % initialize GPU (will erase any existing GPU arrays)
end

[rez, DATA, uproj] = preprocessData2(ops); % preprocess data and extract spikes for initialization
fprintf('Filtering Completed: %0.2f\n', toc)
rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
fprintf('Template Fit Completed: %0.2f\n', toc)
rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)
fprintf('Spike Extraction Completed: %0.2f\n', toc)
% remove temporary file
delete(ops.fproc);
fprintf('Time to process file: %0.2f\n',toc)

return rez
