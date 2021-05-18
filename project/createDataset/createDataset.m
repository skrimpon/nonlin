%% Generate the dataset for 
%
% Authors:  Panagiotis Skrimponis
%           Mustafa Ozkoc
%
% Description: 
clc;
%% Packages
% Add the folder containing +mmwsim to the MATLAB path.
addpath('../mmwComm/');

%% Parameters
fc = 140e+09;       % carrier frequency in Hz
fs = 491.52e6/4;    % sample frequency in Hz
nx = 1e4;           % num of samples
nrx = 16;           % num of RX antennas
nit = 10;           % num of datasets to generate
xvar = 1;           % variance of the TX symbols
isLinear = false;   % 'false' to include the distortion from the rffe
isSave = true;      % 'true' to save the dataset
nbits = 4;          % ADC resolution (i.e., 4-bit). For inf-bit use 0.

txSymType = 'QAM';          % Transmit symbol type: 'iidGaussian', 
                            %                       'iidPhase' or 'QAM'
                            %                       
chanType = 'iidPhase';      % Channel type: 'iidGaussian', 'iidPhase',
                            %               'iidAoA' or 'ones'

% Load the RFFE models
load('rffe140GHz.mat');

% Select components for the LNA, Mixer, LO and ADC.
ilna = 3;
imix = 5;
iplo = 7;
irx = 1;

% Create a receiver based on this configuration.
rx = Rx(...
    'nrx', nrx, ...
    'nx', nx, ...
    'lnaNF', lnaNF(ilna), ...
    'lnaGain', lnaGain(ilna), ...
    'lnaPower', lnaPower(ilna), ...
    'lnaAmpLut', lnaAmpLut(:,:,ilna), ...
    'mixNF', mixNF(iplo,imix), ...
    'mixPLO', mixPLO(iplo), ...
    'mixGain', mixGain(iplo,imix), ...
    'mixPower', mixPower(imix), ...
    'mixAmpLut', reshape(mixAmpLut(:,:,iplo,imix),31,3), ...
    'fs', fs, ...
    'isLinear', isLinear, ...
    'nbits', nbits);

% Calculate the noise floor of the receiver
NF = rx.nf();
T = 290;                            % Ambient temperature in K
k = physconst('Boltzman');          % Boltzmann constant in J/K
noiseFloor = 10*log10(k*T*fs)+NF;   % Noise floor in dB
disp(['Receiver noise floor: ' num2str(noiseFloor+30,'%2.1f') ' dBm'])

snrInTest = 0:5:90;
nsnr = length(snrInTest);

rx.set('snrInTest', snrInTest);

% Create the transmitter object
tx = Tx('nx', nx, 'txSymType', txSymType, 'xvar', xvar);

% Create the channel object
ch = Chan('nx', nx, 'nrx', nrx, 'fs', fs, 'chanType', chanType, 'noiseTemp', T);

%% Generate the dataset
if isSave
    fname = sprintf('../../datasets/rx_%d', irx);
    if isfolder(fname)
        rmdir(fname)
    end
    mkdir(fname)
end
%%
snrOut = zeros(nsnr, nit);
for it = 1:nit
    tic;
    x = tx.step();
    [y, w] = ch.step(x);
    snrOut(:,it) = rx.step(x, y, w);
    if isSave
        T = table;
        T.x = x;
        T.y = y;
        T.w = w;
        T.yant = rx.yant;
        T.yrffe = rx.yrffe;
        T.pwrIn = rx.pwrIn;
        T.pwrOut = rx.pwrOut;
        writetable(T, sprintf('../../datasets/rx_%d/dataset_%d.csv', irx, it));
    end
    toc;
end
snrOut = mean(snrOut, 2);

%% Plot the Output SNR and the RFFE Output Power
rxPwr = reshape(mean(mean(rx.pwrIn, 1), 2), [], 1);
outPwr = reshape(mean(mean(rx.pwrOut, 1), 2), [], 1);
    
figure(1);
clf;
yyaxis left
plot(rxPwr, snrOut, '-o', 'linewidth', 1.5, 'markersize', 5);
box on;
axis tight;
xlabel('Receive power per antenna [dBm]', 'interpreter', 'latex', ...
	'fontsize', 13);
ylabel('Output SNR $\;(\gamma_\mathrm{out})\;$ [dB]', ...
	'interpreter', 'latex', ...
	'fontsize', 13);
grid minor;
ylim([0,33]);

yyaxis right
plot(rxPwr, outPwr, '-^', 'linewidth', 1.5,'markersize', 5);
box on;
ylabel('Output RFFE power [dBm]', ...
	'interpreter', 'latex', ...
	'fontsize', 13);
ylim([-75,-10]); 