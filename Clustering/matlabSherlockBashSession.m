function matlabSherlockBashSession(task, tableFile)

fullTablePath = fullfile(pwd,'TasksDir',tableFile);
if exists(fullTablePath)
    TaskTable = jsondecode(fileread(fullTablePath));
else
    error('Task table not found');
    quit();
end

taskID = strcat('x',num2str(task));
session = TaskTable.(taskID).session_name;
nFiles = TaskTable.(taskID).nFiles;

for f =1:nFiles
    fID = strcat('x',num2str(f));
    fInfo =  TaskTable.(taskID).Files.(fID);
    type = fInfo.type;
    fn = fInfo.filenames;
    sp = fInfo.sp;
    
    if strcmp(type,'KiloSortCluster')
        try
            KiloSort_Master(fn,sp);      
        catch
            disp('Error Running KiloSort');
        end
    end
end
