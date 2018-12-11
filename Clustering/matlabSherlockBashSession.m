function matlabSherlockBashSession(task, tableFile)

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
    fprintf('fID: %s\n',fn);
    sp = fInfo.sp;
    
    if strcmp(type,'KiloSortCluster')
        KiloSort_Master(fn,sp);
       % try
       %     KiloSort_Master(fn,sp);      
       % catch e
       %     disp('Error Running KiloSort');
       %     disp(e.message);
       % end
    end
    fprintf('Clustering Completed for %i\n\n',f);
end
