function CSC_Filt(fn,info)

try
    [~, ~, ~, ~, Samples, Header] = Nlx2MatCSC(fn, [1 1 1 1 1], 1, 1,[]);
    header = extractHeaderInfo(Header);
    signal = Samples(:)*header.ADFactor*1e6; % conver from AD units to uV
catch
    warning(strcat('Could not process file ',FileName'));
end


%% Filter

SR = header.SampRate;
y = filtfilt(info.filt_Coef_b,info.filt_Coef_a,signal);
%[Pxx,F] = pwelch(X,WINDOW,NOVERLAP,NFFT,Fs) returns a PSD computed as
%    a function of physical frequency.  Fs is the sampling frequency
%    specified in hertz.  If Fs is empty, it defaults to 1 Hz.
%pwelch([signal,y],1024,[],4096,32000)
%% Reject Samples based on amplitude
rawSigAmprejectThr = 0.975*header.InputRange;

AmpBadSamps  = abs(signal)>rawSigAmprejectThr;
blurringFilt = ones(round(SR/400),1);
badAmpSampMask  = filtfilt(blurringFilt,1,double(AmpBadSamps))>0;
fprintf('Amplitude Rejection of %.3f pct of data.\n', 100*mean(badAmpSampMask))

header.AmpRejThr = rawSigAmprejectThr;
header.nBadAmpSamps = sum(badAmpSampMask);
y(badAmpSampMask)=nan;
%% Rejection based on slope
% Get derivative of filtered signal, and look for periods of high/low
% variance.
thrDev = 2;
header.slopeDevThr = thrDev;
header.slopeEvalLength = round(SR/100);

dy=diff(y);
if exist('movmad','builtin')==5
    sigDev=movmad(dy,header.slopeEvalLength,'omitnan');
else
    % use custom version of moving average median absolut deviation
    sigDev=movmad2(dy,header.slopeEvalLength);
end
mSigDev=median(sigDev);
sSigDev =std(sigDev);

slopeBadSamps  = abs(sigDev-mSigDev)>=thrDev*sSigDev;
fprintf('Amp and Slope Data Rejection of %.3f pct of data.\n', 100*mean([0;slopeBadSamps]|badAmpSampMask));

header.nSlopeBadSamps = sum(slopeBadSamps);
y(slopeBadSamps)=nan;
%
% figure; hold on;
% hist(sigDev,100)
% plot([1 1]*median(sigDev),ylim,'k','LineWidth',3);
% plot([-2 -2]*sSigDev+mSigDev,ylim,'r','LineWidth',2);
% plot([2 2]*sSigDev+mSigDev,ylim,'r','LineWidth',2);
% xlabel('mad values')
% ylabel('counts')

%%  Normalize signal  (zscore) and perform final rejection
% compute robust std val.
sy = diff(quantile(y,[0.31 0.69]));
zy = (y-nanmedian(y))./sy;

% Final Rejection Step.
% if signal is 10sd beyond mean. exclude those samples.
badAmpSamps2 = abs(zy)>10;
zy(badAmpSamps2) = nan;

header.nBadAmpSamps2 = sum(badAmpSamps2);
header.nTotalRejectedSamps = sum([0;slopeBadSamps]|badAmpSampMask|badAmpSamps2);
header.PctRejectedSamps = 100*mean([0;slopeBadSamps]|badAmpSampMask|badAmpSamps2);
fprintf('Final Rejection of %.3f pct of data.\n', header.PctRejectedSamps);

%% Save
saveFileName = strcat(info.FileName_Head,'_filt.dat');

try
    f = fopen(fullfile(info.savePath,saveFileName),'w');
    fwrite(f,zy,'single');
    fclose(f);
catch
    fprintf('Error. Could not save file\n');
end
try
    saveHeaderFile = strcat(info.FileName_Head,'_header.csv');
    t = struct2table(header);
    writetable(t,fullfile(info.savePath,saveHeaderFile))
catch
    fprintf('Error. Could not save header\n');
end