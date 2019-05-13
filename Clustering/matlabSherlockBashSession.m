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
      if strcmp(type,'KiloSortCluster')
        KiloSort_Master(fn,hfn,sp);
      end
      fprintf('Clustering Completed for %i\n\n',f);
    else
      fprintf('Cluster File %i Exists and Overwrite = False.\n\n',f);
    end
    fn_s = strcat('tt_',str(f),'.bin')
    if ~exists(fullfile(sp,fn_s))
      copyfile(fn,sp)
    end
end
