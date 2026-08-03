%% Moehlis held-out test-data generator
% Generates a held-out set of time series for use as the seed/reference
% input required by predict_using_mlp.py and predict_using_lstm.py
% (dataFilename = "moehlis_test_data_100.mat").
%
% This script is identical to moehlis_data_gen.m in its ODE integration and
% laminarization filter; it differs only in the RNG seed and output
% filename, so the generated series are statistically independent from any
% training set produced by moehlis_data_gen.m.
%
% Output:
%   moehlis_test_data_###.mat
%
% The code has been used for the results in:
% "Predictions of turbulent shear flows using deep neural networks"
% P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa
% Physical Review Fluids (accepted)
%%

% Seed the RNG so this test set never coincides with a training run of
% moehlis_data_gen.m (which does not seed the RNG at all).
rng(12345);

% Number of time series in the output file
nTS = 100;

% Number of timepoints
nTP = 4000;

% Time interval between the timepoints
dt = 1;

%% Parameters
% Reynolds number
Re = 400;

% Size of the domain
Lx = 4*pi;
Lz = 2*pi;

global A B C k1 k2 k3

A = 2*pi/Lx;
B = pi/2;
C = 2*pi/Lz;

k1 = sqrt(A^2 + C^2);
k2 = sqrt(B^2 + C^2);
k3 = sqrt(A^2 + B^2 + C^2);

%%
% Initialize empty 3D matrix for storing data
data = zeros(nTS, nTP, 9);

% Initial conditions
init = [1 0.07066 -0.07076 0 0 0 0 0 0];

count = 1;
while count <= nTS
    disp(count)

    % Add a random perturbation to init(4)
    init(4) = 0.1*rand;

    % Solve ODE
    [t,a_] = ode15s(@(t,a) moehlis_model_odefun(t,a,Re), 0:dt:nTP*dt+99, init);

    % Take only the last nTP points
    a_ = a_(end-nTP+1:end, :);

    % Check for laminarization and add to data matrix only if not
    ind = find(abs(a_(:,1)-1) < 0.01, 1);
    if isempty(ind)
        data(count,:,:) = a_;
        count = count + 1;
    end
end

save(['./moehlis_test_data_' num2str(nTS) '.mat'], 'data')
