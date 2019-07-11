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
    try
      fID = strcat('x',num2str(f));
      fInfo =  TaskTable.(taskID).Files.(fID);
      type = fInfo.type;
      fn = fInfo.filenames;
      hfn = fInfo.headerFile;
      fprintf('fID: %s\n',fn);
      sp = fInfo.sp;

      if (~exist(fullfile(sp,'rez.mat')) || overwriteFlag)
        if strcmp(type,'KiloSortTTCluster')
          rez=KiloSort_Master(fn,hfn,sp,'tt');
        else if strcmp(type,'KiloSortNR32Cluster')
          rez=KiloSort_Master(fn,hfn,sp,'NR32');
        end
        % save matlab results file
        save(fullfile(ops.root,  'rez.mat'), 'rez', '-v7.3');
        % save python results file for Phy
        rezToPhy(rez, ops.root);
        fprintf('Clustering Completed for %i\n\n',f);

        if strcmp(type,'KiloSortTTCluster')
          fn_s = strcat('tt_',str(f),'.bin')
        else
          fn_s = strcat('probe.bin')
        end
        if ~exists(fullfile(sp,fn_s))
          copyfile(fn,sp)
        end

        sp2 = fInfo.sp2;
        if length(sp2)>0
          if ~exists(fullfile(sp2,fn_s))
            save(fullfile(sp2,  'rez.mat'), 'rez', '-v7.3');
            rezToPhy(rez, sp2);
            copyfile(fn,sp2);
          end
        end

      else
        fprintf('Cluster File %i Exists and Overwrite = False.\n\n',f);
      end
    catch
      fprintf('Error Processing File %i\n',f)
    end
end
