
clc;
%% Packages
% Add the folder containing +mmwsim to the MATLAB path.
addpath('../mmwComm/');

%%
ilna = 3;
imix = 5;
iplo = 7;
nx = 16384;
nrx = 16;
nbits = 5;
dither = false;
chanType = 'iidPhase';
txSymType = 'iidGaussian';
% Load the RFFE models
load('rffe140GHz.mat');
%%
% fs = 1.96608e+09;   % sample frequency in Hz
fs = 491.52e6/4;
% fs = 100e6;
% fs = nx;   % sample frequency in Hz
NF = 10*log10(10^(0.1*lnaNF(ilna)) + 10^(-0.1*lnaGain(ilna))*(10^(0.1*mixNF(iplo,imix))-1));
T = 290;        % Ambient temperature (K)
BW = fs;        % Bandwidth (Hz)
k =  physconst('Boltzman'); % Boltzmann constant (J/K)
noiseFloor = 10*log10(k*T*BW)+NF; % dB
disp(['Receiver noise floor: ' num2str(noiseFloor+30,'%2.1f') ' dBm'])
%%

agc0 = comm.AGC;
agc = MultiInput(agc0, nrx);

SampleRate = fs;
ReferenceImpedance = 1;
tn0 = comm.ThermalNoise(...
    'SampleRate', SampleRate, ...
    'NoiseTemperature', T);
tn = MultiInput(tn0, nrx);

lnaNoise0 = comm.ThermalNoise(...
    'NoiseMethod', 'Noise figure', ...
    'NoiseFigure', lnaNF(ilna), ...
    'SampleRate', SampleRate);
lnaNoise = MultiInput(lnaNoise0, nrx);

lnaAmp0 = comm.MemorylessNonlinearity(...
    'Method', 'Lookup table', ...
    'Table', lnaAmpLut(:,:,ilna), ...
    'ReferenceImpedance', ReferenceImpedance);
lnaAmp = MultiInput(lnaAmp0, nrx);

mixNoise0 = comm.ThermalNoise(...
    'NoiseMethod', 'Noise figure', ...
    'NoiseFigure', 10*log10(10^(-0.1*lnaGain(ilna))*(10^(0.1*mixNF(iplo,imix))-1)+1), ...
    'SampleRate', SampleRate);
mixNoise = MultiInput(mixNoise0, nrx);

mixAmp0 = comm.MemorylessNonlinearity(...
    'Method', 'Lookup table', ...
    'Table', reshape(mixAmpLut(:,:,iplo,imix), 31, 3), ...
    'ReferenceImpedance', ReferenceImpedance);
mixAmp = MultiInput(mixAmp0, nrx);

% Baseband AGC used to adjust the input level to the ADC-
% This would be performed via a controllable baseband amplifier
bbAGC = mmwsim.rffe.AutoScale('meth', 'MatchTgt');

% ADC
adc = mmwsim.rffe.ADC( ...
    'nbits', nbits, ...
    'dither', dither, ...
    'outputType', 'int');
% adc.optScale();

% Find optimal input target for the ADC
% Full scale value
adcFS = max(adc.stepSize*(2^(nbits-1)-1), 1);
EsFS = 2*adcFS^2;

% Test the values at some backoff from full scale
bkfTest = linspace(-30,5,100)';
EsTest = EsFS*10.^(0.1*bkfTest);

% Compute the SNR
snr = bbAGC.compSnrEs(EsTest, adc);

% Select the input level with the maximum SNR
[~, im] = max(snr);
bbAGC.set('EsTgt', EsTest(im));
%%
awgnChannel = comm.AWGNChannel(...
    'NoiseMethod', 'Variance', ...
    'Variance', 10^(0.1*noiseFloor));

% Create the transmitter object
tx = Tx('nx', nx, 'txSymType', txSymType);

% Create the channel object
ch = Chan('nx', nx, 'nrx', nrx, 'chanType', chanType, 'noiseTemp', T);

for it2 = 1:10
    tic;
    x = tx.step();
    [y, w] = ch.step(x);
    y = y./sqrt(mean(abs(y).^2, 'all'));

    testInputLevelOffsets = -10:2:80'; % dB
    testInputLevels = noiseFloor+testInputLevelOffsets+30; % dBm
    A = 10.^((testInputLevels-30)/20);      % Voltage gain (attenuation)
    % A = A*sqrt(nrx);                        % Account for generator scaling
    nit = size(A,2);
    snrOut = zeros(nit,1);
    rxPwr = zeros(nit,1);
    outPwr = zeros(nit,1);
    nsnr = length(testInputLevelOffsets);
    yIn = zeros(nx, nrx, nsnr);
    yOut = zeros(nx, nrx, nsnr);
    pwrIn = zeros(nx, nrx, nsnr);
    pwrOut = zeros(nx, nrx, nsnr);

    for it = 1:nit
        yrx = y.*A(it);
        pwrIn(:,:,it) = 10*log10(yrx.*conj(yrx))+30;
        % Measure the average power at the antenna connector in Watts
        measuredPower = mean(yrx.*conj(yrx), 'all');
        rxPwr(it) = 10*log10(measuredPower)+30;
        r2 = awgnChannel(yrx);
        r3 = mixNoise(lnaNoise(tn(yrx)));
        r = mixAmp(mixNoise(lnaAmp(lnaNoise(tn(yrx)))));
        % r = (mixNoise((lnaNoise(tn(yrx)))));
        % r = r3;
        % fprintf('%.4f %.4f\n', 10*log10(mean(r2.*conj(r2), 'all')), 10*log10(mean(r3.*conj(r3), 'all')))
        measuredPower = mean(mean(r.*conj(r)));
        outPwr(it) = 10*log10(mean(measuredPower))+30;
        pwrOut(:,:,it) = 10*log10(r.*conj(r))+30;

        r = adc(bbAGC(r));
        xhat = sum(r.*conj(w),2) ./ sum(abs(w).^2,2);
        a = mean(conj(xhat).*x)/mean(abs(x).^2);
        dvar = mean(abs(xhat - a*x).^2);
        snrOut(it) = 10*log10(abs(a).^2/dvar);
        yIn(:,:,it) = yrx;
        yOut(:,:,it) = r;
    end

    T = table;
    T.x = x;
    T.y = y;
    T.w = w;
    T.yant = yIn;
    T.yrffe = yOut;
    T.pwrIn = pwrIn;
    T.pwrOut = pwrOut;
    writetable(T, sprintf('../../datasets/new/dataset_%d.csv', it2));
    toc;
end

figure(1);
clf;
yyaxis left
plot(rxPwr, snrOut, '-o', 'linewidth', 1.5);
box on;
axis tight;
xlabel('Receive power per antenna [dBm]', 'interpreter', 'latex', ...
	'fontsize', 13);
ylabel('Output SNR $\;(\gamma_\mathrm{out})\;$ [dB]', ...
	'interpreter', 'latex', ...
	'fontsize', 13);
grid on;

yyaxis right
plot(rxPwr, outPwr, '-x', 'linewidth', 1.5);
box on;
axis tight;
ylabel('Output RFFE power [dBm]', ...
	'interpreter', 'latex', ...
	'fontsize', 13);
grid on;