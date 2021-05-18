classdef Chan < matlab.System
    % Chan: 
    
    properties
        nx = 1e4;       % num of samples
        nrx = 16;       % num of RX antennas
        chanType;       % Channel type: 'iidGaussian', 'iidPhase', ...
                        %               'iidAoA' or 'ones'
        noiseTemp;      % noise temperature in K
        EkT;            % thermal noise energy
        fs;
    end
    
    methods
        function obj = Chan(varargin)
            % Constructor
            
            % Set parameters from constructor arguments.
            if nargin >= 1
                obj.set(varargin{:});
            end
            
            obj.EkT = physconst('Boltzman')*obj.noiseTemp;
        end
    end
    methods (Access = protected)
        function [y, w] = stepImpl(obj, x)
            % Generate a random channel
            if strcmp(obj.chanType,'iidPhase')
                phase = 2*pi*rand(obj.nx,obj.nrx);
                w = exp(1j*phase);
            elseif strcmp(obj.chanType, 'iidGaussian')
                w = (randn(obj.nx,obj.nrx) + 1j*randn(obj.nx,obj.nrx))/sqrt(2);
            elseif strcmp(obj.chanType, 'randAoA')
                dsep = 0.5;
                theta = unifrnd(-pi/2,pi/2,obj.nx,1);
                phase = 2*pi*cos(theta)*(0:obj.nrx-1)*dsep;
                w = exp(1j*phase);
            elseif strcmp(obj.chanType, 'ones')
                w = ones(obj.nx,obj.nrx);
            else
                error('Unknown channel type');
            end
            
            % Generate RX symbols with no noise
            y = x.*w;
            
            % Rescale so that it is Es/(fs*kT) = 1
            scale = sqrt((obj.fs*obj.EkT)/mean(abs(y).^2, 'all'));
            y = y * scale;
        end
    end
end