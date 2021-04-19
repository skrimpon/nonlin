classdef Rx < matlab.System
    % RX:
    
    properties
        nx = 1e4;       % num of samples
        nrx = 16;       % num of RX antennas
        xvar = 1;       % variance of the TX symbols
        
        % RFFE
        rffe;           % RFFE object
        lnaGain;        % lna gain in dB
        lnaNF;			% lna noise figure in dB
        lnaAmpLut;      % lna fund tone power curve
        lnaPower;       % lna power in mW
        
        mixGain;        % mixer gain in dB
        mixNF;          % mixer noise figure in dB
        mixAmpLut;      % mixer fund tone power curve
        mixPLO;         % mixer input power from the local oscillator
        mixPower;       % mixer power in mW
        
        fs;             % sampling frequency in Hz
		snrInTest;      % input SNR
        
        isLinear;       % `true` to include the rffe non-linearities
        nbits;          % num of ADC bits
        
        designID;       % receiver design ID
        yant;
        yrffe;
        xhat;
    end
    
    properties (Dependent)
        isLinearUpdate; % `true` to include the rffe non-linearities
        nbitsUpdate;    % num of ADC bits
    end
    
    methods
        function obj = Rx(varargin)
            % Constructor
            
            % Set parameters from constructor arguments.
            if nargin >= 1
                obj.set(varargin{:});
            end
            
            % Create the RX RFFE
            obj.rffe = mmwsim.rffe.RFFERx(...
                'nrx', obj.nrx, ...
                'lnaNF', obj.lnaNF, ...
                'lnaGain', obj.lnaGain, ...
                'lnaPower', obj.lnaPower, ...
                'lnaAmpLut', obj.lnaAmpLut, ...
                'mixNF', obj.mixNF, ...
                'mixPLO', obj.mixPLO, ...
                'mixGain', obj.mixGain, ...
                'mixPower', obj.mixPower, ...
                'mixAmpLut', obj.mixAmpLut, ...
                'fsamp', obj.fs, ...
                'nbits', obj.nbits, ...
                'isLinear', obj.isLinear);
        end
        
        % Create some helper functions
        function set.isLinearUpdate(obj, val)
            obj.isLinear = val;
            obj.rffe.set('isLinear', val);
        end
        
        function set.nbitsUpdate(obj, val)
            obj.nbits = val;
            obj.rffe.set('nbits', val);
            obj.rffe.elem{end}.set('nbits', val);
        end
        
		function NF = nf(obj)
			% Calculate the effective noise figure
			NF = obj.rffe.nf();
		end
		
		function P = power(obj)
			P = obj.rffe.power();
		end
    end
    
    methods (Access = protected)
        function [snrOut] = stepImpl(obj, x, y, w)
            % Find the number of rf elements
            nsnr = length(obj.snrInTest);
            snrOut = zeros(nsnr, 1);
            
            obj.yant = zeros(obj.nx, obj.nrx, nsnr);
            obj.yrffe = zeros(obj.nx, obj.nrx, nsnr);
            obj.xhat = zeros(obj.nx, nsnr);
            for isnr = 1:nsnr
                % Get the SNR and scale the input signal
                obj.yant(:,:,isnr) = 10^(0.05*obj.snrInTest(isnr))*y;
                
                % Run through RFFE stages
                obj.yrffe(:,:,isnr) = obj.rffe.step(obj.yant(:,:,isnr));
                
                % Add thermal noise
                obj.yant(:,:,isnr) = obj.rffe.elem{1}.step(obj.yant(:,:,isnr));
                
                % Use the known channel w to beamform
                obj.xhat(:,isnr) = sum(obj.yrffe(:,:,isnr).*conj(w),2) ./ sum(abs(w).^2,2);
                
                % Consider a linear estimate to find xhat
                %
                % xhat = a*x + d,  d ~ CN(0, E|xhat-x|^2)
                a = mean(conj(obj.xhat(:,isnr)).*x)/mean(abs(x).^2);
                dvar = mean(abs(obj.xhat(:,isnr) - a*x).^2);
                
                % Measure the output SNR
                snrOut(isnr) = 10*log10(abs(a).^2*obj.xvar/dvar);
            end
        end
    end
end

