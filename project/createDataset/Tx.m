classdef Tx < matlab.System
    % Tx:
    
    properties
        nx = 1e4;                   % num of samples
        txSymType = 'iidGaussian';  % Transmit symbol type: 'iidGaussian' 
                                    %                        or 'iidPhase'
        xvar = 1;                   % variance of the tx symbols
    end
    
    methods
        function obj = Tx(varargin)
            % Constructor
            
            % Set parameters from constructor arguments.
            if nargin >= 1
                obj.set(varargin{:});
            end
        end
    end
    
    methods (Access = protected)
        function [x] = stepImpl(obj)
            % Generate random symbols without scaling
            if strcmp(obj.txSymType, 'iidGaussian')
                x = (randn(obj.nx,1) + 1j*randn(obj.nx,1))*sqrt(obj.xvar/2);
            elseif strcmp(obj.txSymType, 'iidPhase')
                x = exp(1j*2*pi*rand(obj.nx,1));
            elseif strcmp(obj.txSymType, 'QAM')
                bit = randi([0, 1], log2(4), obj.nx);
                x = qammod(bit, 4, 'InputType', 'bit', 'UnitAveragePower', true);
                x = x';
            else
                error('Unknown TX symbol type');
            end
        end
    end
end

