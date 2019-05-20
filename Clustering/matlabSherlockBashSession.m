function matlabSherlockBashSession(task, tableFile,overwriteFlag)

fullTablePath = fullfile(pwd,'TasksDir',tableFile);
if exist(fullTablePath)
    TaskTable = jsondecode(fileread(fullTablePath));
else
    error('Task table not found');
    quit();
end

taskID = strcat('x',num2str(task));
session = TaskTable.(taskID).session_name;
nFiles = TaskTable.(taskID).nFiles;

for f =1:nFiles
    fprintf('Processing File # %i \n',f);
    fID = strcat('x',num2str(f));
    fInfo =  TaskTable.(taskID).Files.(fID);
    type = fInfo.type;
    fn = fInfo.filenames;
    hfn = fInfo.headerFile;
    fprintf('fID: %s\n',fn);
    sp = fInfo.sp;

    if ~exist(fullfile(sp,'rez.mat')) | overwriteFlag
      if strcmp(type,'KiloSortTTCluster')
        KiloSort_Master(fn,hfn,sp,'tt');
      else if strcmp(type,'KiloSortNR32Cluster')
        KiloSort_Master(fn,hfn,sp,'NR32');
      end
      fprintf('Clustering Completed for %i\n\n',f);
    else
      fprintf('Cluster File %i Exists and Overwrite = False.\n\n',f);
    end
    if strcmp(type,'KiloSortTTCluster')
      fn_s = strcat('tt_',str(f),'.bin')
    else
      fn_s = strcat('probe.bin')
    end
    if ~exists(fullfile(sp,fn_s))
      copyfile(fn,sp)
    end
end
