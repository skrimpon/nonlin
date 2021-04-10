%%
%
% Description:


%% Packages
% Add the folder containing +mmwsim to the MATLAB path.
addpath('../../mmwComm/');

%% Parameters
fc = 140e+09;       % carrier frequency in Hz
fs = 1.96608e+09;   % sample frequency in Hz
nx = 1e4;           % num of samples
nrx = 16;           % num of RX antennas
nit = 5;            % num of iterations
nsnr = 31;          % num of SNR points
xvar = 1;           % variance of the TX symbols
ndrivers = length(0:log2(nrx)); % num of LO driver configurations

txSymType = 'iidGaussian';  % Transmit symbol type: 'iidGaussian' or
                            %                       'iidPhase'
chanType = 'iidPhase';      % Channel type: 'iidGaussian', 'iidPhase',
                            %               'iidAoA' or 'ones'

% Load the RFFE models
load('rffe140GHz.mat');

% Input SNR Es/N0 relative to thermal noise
snrInTest = linspace(-10, 50, nsnr)';

% Compute received input power Pin in dBm
noiseTemp = 290;						% noise temperature in K
EkT = physconst('Boltzman')*noiseTemp;	% noise energy
Pin = 10*log10(EkT*fs) + 30 + snrInTest;

%% Run the simulations
%
% 1. Design 1
% 2. Design 1 w/o A/D
% 3. Design 1 w/ Linear RFFE
% 4. Design 1 w/o A/D w/ Linear RFFE
% 5. Design 2
% 6. Design 2 w/o A/D,
% 7. Design 2 w/ Linear RFFE
% 8. Design 2 w/o A/D w/ Linear RFFE
designID = [1;1;1;1;2;2;2;2];
ndsgn = length(unique(designID));
for idsgn = 1:ndsgn
    fname = sprintf('../../datasets/rx_%d',idsgn);
%    rmdir(fname);   % delete previous data folder
    mkdir(fname);   % create a new folder
end
adcTest = [4;0;4;0;5;0;5;0];
lnaTest = [4;4;4;4;3;3;3;3];
mixerTest = [5;5;5;5;5;5;5;5];
ploTest = [2;2;2;2;7;7;7;7];
linTest = [false; false; true; true; false; false; true; true];
nsim = length(adcTest);

% Intialize vectors
snrOut = zeros(nsnr, nsim, nit);
tx = cell(nit,1);
ch = cell(nit,1);
rx = cell(nsim, nit);

for it = 1:nit
    % Create the transmitter object
    tx{it} = Tx('nx', nx, 'txSymType', txSymType);
    
    % Create the channel object
    ch{it} = Chan('nx', nx, 'nrx', nrx, 'chanType', chanType, 'noiseTemp', noiseTemp);
    
    for isim = 1:nsim
        rx{isim,it} = Rx(...
            'nrx', nrx, ...
            'lnaNF', lnaNF(lnaTest(isim)), ...
            'lnaGain', lnaGain(lnaTest(isim)), ...
            'lnaPower', lnaPower(lnaTest(isim)), ...
            'lnaAmpLut', lnaAmpLut(:,:,lnaTest(isim)), ...
            'mixNF', mixNF(ploTest(isim)), ...
            'mixPLO', mixPLO(ploTest(isim)), ...
            'mixGain', mixGain(mixerTest(isim)), ...
            'mixPower', mixPower(mixerTest(isim)), ...
            'mixAmpLut', reshape(mixAmpLut(:,:,ploTest(isim),mixerTest(isim)),31,3), ...
            'fs', fs, ...
            'isLinear', linTest(isim), ...
            'nbits', adcTest(isim), ...
            'snrInTest', snrInTest, ...
            'designID', designID(isim));
    end
end


for it = 1:nit
    tic;
    % Generate new data
    x = tx{it}.step();
    
    % Send the data over the channel
    [y, w] = ch{it}.step(x);
    
    % Receive data
    parfor isim = 1:nsim
        snrOut(:,isim,it) = rx{isim,it}.step(x, y, w);
        
        % Save output data
        T = table;
        T.yrffe = rx{isim,it}.yrffe;
        T.xhat = rx{isim,it}.xhat;
        
        writetable(T, sprintf('../../datasets/rx_%d/odata_%d_%d.csv', ...
            rx{isim,it}.designID, isim, it));
    end    
    
    % Save common data
    T = table;
    T.x = x;
    T.y = y;
    T.w = w;
    tmp = rx{1,it}.step(x, y, w);
    T.yant = rx{1,it}.yant;
    
    for idsgn = 1:ndsgn
        writetable(T, sprintf('../../datasets/rx_%d/idata_%d.csv', ...
            designID(idsgn), it));
    end
    toc;
end

% Average over all iterations.
snrOut = mean(snrOut, 3);

% Calculate the saturation SNR
snrSat = snrOut(end, :)';

%% Find the Effective Noise Figure and Power Consumption

rffePower = zeros(nsim, ndrivers);
rffeNF = zeros(nsim, 1);

for isim = 1:nsim
    rffePower(isim, :) = rx{isim}.power();	% RFFE power consumption [mW]
    rffeNF(isim) = rx{isim}.nf();			% Effective noise figure [dBm]
end

% Find the minimum power for each parameter setting. The `idriver` will
% denote the number of LO drivers.
[rffePower, idriver] = min(rffePower, [], 2);

%% Fit a model
% We use two parameters to characterize the performance of each
% configuration: (a) effective noise figure that is dominant in low input
% power; (b) saturation SNR that is dominant in high SNR. Using these
% values we can fit a model for the output SNR as follows,

% For non-linear systems we empirically show that the output SNR can be
% calculated by the following formula
nom = nrx * 10.^(0.1*snrInTest);
denom = reshape(10.^(0.1*rffeNF), [], nsim) + ...
    nrx * reshape(10.^(0.1*snrInTest), nsnr, []) .* ...
    reshape(10.^(-0.1*snrSat), [], nsim);
rffeModel = 10*log10(nom./denom);

%% Save the paramters for each receiver design

it = 1;
tic;
for isim = 1:nsim
    T = table;
    T.fc = fc;
    T.fs = fs;
    T.nx = nx;
    T.nit = nit;
    T.nrx = nrx;
    T.nsnr = nsnr;
    T.xvar = xvar;
    T.idriver = idriver(isim);
    T.nbits = adcTest(isim);
    T.isLinear = linTest(isim);
    T.txSymType = tx{it}.txSymType;
    T.chanType = ch{it}.chanType;
    T.rffePower = rffePower(isim);
    T.snrSat = snrSat(isim);
    
    writetable(T, sprintf('../../datasets/rx_%d/param_0_%d_%d.csv', ...
        rx{isim,it}.designID, isim, it));
    
    T = table;
    T.snrOut = snrOut(:,isim);
    T.rffeModel = rffeModel(:,isim);
    T.Pin = Pin;
    
    writetable(T, sprintf('../../datasets/rx_%d/param_1_%d_%d.csv', ...
        rx{isim,it}.designID, isim, it));
end
toc;