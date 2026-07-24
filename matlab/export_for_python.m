% export_for_python.m
% Run from the matlab/ directory: addpath(pwd); export_for_python
% Saves all intermediate MMM data as .mat files in ../test_data/

outdir = '../test_data';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

%% Parameters (matching AxiHorndemo1) =====================================
sth = 10e-4; sm = 500e-4; Lh = 30e-2; dz = Lh/250;
T = 1;
HornType = 'exponential';
rth = sqrt(sth/pi); rm = sqrt(sm/pi);

fmin = 100; fmax = 15000; Nf = 200; N = 8;
c = 344; rho = 1.205;
freq = logspace(log10(fmin), log10(fmax), Nf);

Rext = 3;
Angext = linspace(0, 90, 181)';

%% 1. MMM_1Dhorncoord ====================================================
horncoords = MMM_1Dhorncoord(HornType, rth, rm, Lh, T, dz, false, 0, 0.1, 80);
save(fullfile(outdir, 'horncoords.mat'), ...
    'horncoords', 'HornType', 'rth', 'rm', 'Lh', 'T', 'dz');

%% 2. MMM_init ===========================================================
data = MMM_init(freq, N, horncoords, 'axi', rho, c);
rawCoords = data.rawCoords;
steppedCoords = data.steppedCoords;
eigenValues = data.eigenValues;
bigF = data.bigF;
kvec = data.k;
S = data.S;
Sm = data.Sm;
St = data.St;
nModes = data.nModes;
nfreq = data.nfreq;
save(fullfile(outdir, 'init_data.mat'), ...
    'rawCoords', 'steppedCoords', 'eigenValues', 'bigF', ...
    'kvec', 'S', 'Sm', 'St', 'nModes', 'nfreq', 'freq', 'rho', 'c');

%% 3. MMM_ASbaffledradzmatrixIntp =======================================
Zrad = MMM_ASbaffledradzmatrixIntp(data.k, data.rho, data.c, data.Sm, ...
                                   data.nModes, 'ZradAS32.mat');
save(fullfile(outdir, 'zrad.mat'), 'Zrad');

%% 4. MMM_calculateMatrices ==============================================
data.Zrad = Zrad;
data = MMM_calculateMatrices(data, false);
BigZ = data.BigZ;
Umat = data.Umat;
Z00 = data.Z00;
UmouthPw = data.UmouthPw;
Umouth = data.Umouth;
save(fullfile(outdir, 'calculate.mat'), ...
    'BigZ', 'Umat', 'Z00', 'UmouthPw', 'Umouth');

%% 5. MMM_ASradiatedPressure =============================================
fieldPoints = [Rext * sin(Angext * pi / 180), ...
               Rext * cos(Angext * pi / 180)];
data = MMM_ASradiatedPressure(data, fieldPoints, true);
pRad = data.pRad;
save(fullfile(outdir, 'prad.mat'), 'pRad', 'fieldPoints', 'Angext', 'Rext');

%% 6. MMM_ASgetDI ========================================================
data = MMM_ASgetDI(data, Angext);
DI = data.DI;
save(fullfile(outdir, 'di.mat'), 'DI', 'Angext', 'freq');

%% 7. Isolated unit tests =================================================
load('MMM_besselzeros.mat', 'bz');
bz5 = bz(1:5);

% MMM_ASmakefmat: R1 < R2 (expanding)
F_expand = MMM_ASmakefmat(5, [0 0.01], [0 0.02], bz5);
save(fullfile(outdir, 'makefmat_expanding.mat'), 'F_expand');

% MMM_ASmakefmat: R1 > R2 (contracting)
F_contract = MMM_ASmakefmat(5, [0 0.02], [0 0.01], bz5);
save(fullfile(outdir, 'makefmat_contracting.mat'), 'F_contract');

% MMM_ASmakefmat: R1 == R2 (identity)
F_equal = MMM_ASmakefmat(5, [0 0.01], [0 0.01], bz5);
save(fullfile(outdir, 'makefmat_equal.mat'), 'F_equal');

% MMM_ASmakekm: single k at first duct step
km = MMM_ASmakekm(data.k(1), data.steppedCoords(1, :), N, bz);
save(fullfile(outdir, 'makekm.mat'), 'km');

% MMM_ASgeteigenfunctions
R_mouth = sqrt(data.Sm / pi);
r_test = linspace(0, R_mouth, 20);
phi = MMM_ASgeteigenfunctions(R_mouth, r_test, data.eigenValues(1:N), true);
save(fullfile(outdir, 'eigenfunctions.mat'), 'phi', 'R_mouth', 'r_test');

fprintf('Export complete. Data saved to %s/\n', outdir);
