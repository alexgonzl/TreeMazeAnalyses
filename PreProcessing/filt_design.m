fs = 32000; % sampling rate
Ny = fs/2;  % nyquist
fOrder = 64;

% initial iir filt
[b0,a] = cheby1(3,0.001,300/Ny,'high');

b_lowpass =  fir1(fOrder, [6500]/Ny,kaiser(fOrder+1,8));
b_highpass=  fir1(fOrder*2, [300]/Ny,'high',blackman(fOrder*2+1));

% form equivalent filter
beq=conv(conv(b0,b_lowpass),b_highpass);
beq2=beq;
beq2(abs(beq)<1e-8)=[];
%%
fvtool(beq,a,beq2,a,'fs',fs)

%%
dlmwrite('filt_a.dat',a,'delimiter',',','precision',12)
dlmwrite('filt_b.dat',beq2,'delimiter',',','precision',12)