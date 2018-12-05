function matlabSherlockBashSession(task, table)

fullTablePath = fullfile(pwd,'TasksDir',table);
if exists(fullTablePath)
    TaskTable = jsondecode(fileread(fullTablePath));
else
    error('Task table not found');
end





