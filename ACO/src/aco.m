clc; close all;

if ~exist('seed', 'var') || isempty(seed)
    seed = floor(mod(now * 86400000, 2147483647));
end
if ~exist('run_id', 'var'), run_id = 0; end
if exist('profile', 'var') && ~exist('weight_profile', 'var')
    weight_profile = profile;
end
if ~exist('weight_profile', 'var'), weight_profile = 'A'; end
if ~exist('W', 'var'), W = []; end

if ischar(seed), seed = str2double(seed); end
if ~isfinite(seed)
    seed = floor(mod(now * 86400000, 2147483647));
end
seed = floor(mod(seed, 2147483647));
if seed <= 0, seed = 1; end

if exist('OCTAVE_VERSION', 'builtin') ~= 0
    rand('seed', seed);
    randn('seed', seed);
else
    rng(seed, 'twister');
end

fprintf('[ACO-VNEQTS] seed=%d  run_id=%d\n', seed, run_id);

csvPath = 'bs_statistics_all_12files.csv';
if exist(csvPath, 'file') ~= 2
    error('Cannot find %s. Run bs_v2.cpp first or copy the generated CSV here.', csvPath);
end

model = loadBSModelVNEQTS(csvPath);
[model.W, weightProfileName, weightProfileLabel] = resolveWeightProfileVNEQTS(weight_profile, W);
model.capacity_hard_constraint = true;
model.D_MAX = 4800.0;
model.local_lambda = zeros(model.num_bs, 1);
for i = 1:model.num_bs
    nearMask = model.dist_matrix_m(i, :).' <= model.D_MAX;
    model.local_lambda(i) = sum(model.lambda(nearMask));
end
model.pheremon = model.local_lambda;
N = model.num_bs;
M = model.M_BUDGET;
nMax = model.N_MAX;

% ACO parameters. MaxIt/nAnt can be overridden from the caller workspace.
if ~exist('MaxIt', 'var'), MaxIt = 500; end
if ~exist('nAnt', 'var'),  nAnt = 50; end
if ~exist('alpha', 'var'), alpha = 1; end
if ~exist('beta', 'var'),  beta = 2; end
if ~exist('rho', 'var'),   rho = 0.05; end

tau0 = 1.0;
Q = 1.0;
eliteWeight = 2.0;
tauMin = 1e-6;
tauMax = 100.0;

tau = tau0 * ones(N, 1);

% Heuristic desirability: total edge demand within D_MAX of each candidate site.
eta = model.local_lambda(:);
if max(eta) > 0
    eta = eta ./ max(eta);
end
eta = eta + 0.05;

fprintf('Weight profile: %s, W=[%.2f %.2f %.2f %.2f]\n', ...
    weightProfileName, model.W(1), model.W(2), model.W(3), model.W(4));

convPath = sprintf('convergence_ACO_%s_run%d.csv', weightProfileLabel, run_id);
fidConv = fopen(convPath, 'w');
if fidConv < 0
    error('Cannot open %s for writing.', convPath);
end
fprintf(fidConv, 'Iteration,BestFitness,PhiSite,PhiTraf,AvgUtilization,Balance,EnergyEff,TotalEnergy,CloudFallback,MeanFitness,StdFitness\n');

Best.n = zeros(N, 1);
Best.fitness = -Inf;
Best.metrics = [];
BestFitness = zeros(MaxIt, 1);

fprintf('ACO starts: N=%d, M=%d, nmax=%d, ants=%d, iter=%d\n', ...
    N, M, nMax, nAnt, MaxIt);
tStart = tic;

for it = 1:MaxIt
    ants = repmat(struct('n', [], 'fitness', -Inf, 'metrics', []), nAnt, 1);
    fitThisGen = zeros(nAnt, 1);

    for k = 1:nAnt
        n = constructAcuSolutionACO(tau, eta, M, nMax, alpha, beta);
        metrics = evaluateDeploymentVNEQTS(n, model);

        ants(k).n = n;
        ants(k).fitness = metrics.fitness;
        ants(k).metrics = metrics;
        fitThisGen(k) = metrics.fitness;

        if metrics.fitness > Best.fitness
            Best.n = n;
            Best.fitness = metrics.fitness;
            Best.metrics = metrics;
        end
    end

    tau = (1 - rho) * tau;
    for k = 1:nAnt
        tau = tau + (Q * max(ants(k).fitness, 0) / M) * ants(k).n;
    end
    tau = tau + (eliteWeight * Q * max(Best.fitness, 0) / M) * Best.n;
    tau = min(max(tau, tauMin), tauMax);

    BestFitness(it) = Best.fitness;
    meanFitness = mean(fitThisGen);
    stdFitness = std(fitThisGen, 1);

    fprintf(fidConv, '%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f\n', ...
        it, Best.metrics.fitness, Best.metrics.phi_site, Best.metrics.phi_traf, ...
        Best.metrics.avg_utilization, Best.metrics.balance, Best.metrics.energy_eff, ...
        Best.metrics.total_energy, Best.metrics.cloud_fallback, meanFitness, stdFitness);

    if mod(it, 50) == 0 || it == 1
        fprintf('Iter %d | F=%.6f | Phi_site=%.4f | Phi_traf=%.4f | U=%.4f | B=%.4f | Ee=%.4f | CF=%d\n', ...
            it, Best.metrics.fitness, Best.metrics.phi_site, Best.metrics.phi_traf, ...
            Best.metrics.avg_utilization, Best.metrics.balance, ...
            Best.metrics.energy_eff, Best.metrics.cloud_fallback);
    end
end

elapsed = toc(tStart);
fclose(fidConv);

bestMetrics = Best.metrics;
activeSites = find(Best.n > 0);

acuFile = sprintf('ACO_acu_deployment_%s_run%d.csv', weightProfileLabel, run_id);
fidAcu = fopen(acuFile, 'w');
if fidAcu < 0
    error('Cannot open %s for writing.', acuFile);
end
fprintf(fidAcu, 'BS_ID,Latitude,Longitude,ACU_Count,Capacity,ES_Lambda,ES_Utilization_pct\n');
for p = 1:numel(activeSites)
    i = activeSites(p);
    cap = bestMetrics.capacity(i);
    utilPct = 0.0;
    if cap > 0
        utilPct = min(bestMetrics.load_per_site(i) / cap, 1.0) * 100.0;
    end
    fprintf(fidAcu, '%s,%.6f,%.6f,%d,%.6f,%.6f,%.2f\n', ...
        model.bs_ids{i}, model.lat(i), model.lon(i), Best.n(i), cap, ...
        bestMetrics.load_per_site(i), utilPct);
end
fclose(fidAcu);

allocFile = sprintf('ACO_bs_es_allocation_%s_run%d.csv', weightProfileLabel, run_id);
fidAlloc = fopen(allocFile, 'w');
if fidAlloc < 0
    error('Cannot open %s for writing.', allocFile);
end
fprintf(fidAlloc, 'BS_ID,Latitude,Longitude,Lambda,ACU_Count,Serving_ES_ID,Dist_to_ES_m\n');
for k = 1:N
    esIdx = bestMetrics.sigma(k);
    acuCount = Best.n(k);
    if esIdx > 0
        servingId = model.bs_ids{esIdx};
        distToEs = model.dist_matrix_m(k, esIdx);
    else
        servingId = 'CLOUD';
        distToEs = -1.0;
    end
    fprintf(fidAlloc, '%s,%.6f,%.6f,%.6f,%d,%s,%.1f\n', ...
        model.bs_ids{k}, model.lat(k), model.lon(k), model.lambda(k), ...
        acuCount, servingId, distToEs);
end
fclose(fidAlloc);

multiFile = 'single_run_weight_profiles_ACO_VNEQTS.csv';
multiHeader = 'Algorithm,WeightProfile,RunID,Seed,W_PhiSite,W_PhiTraf,W_Balance,W_EnergyEff,F,PhiSite,PhiTraf,AvgUtilization,Balance,EnergyEff,TotalEnergy,CloudFallback,ActiveSites';
if exist(multiFile, 'file') == 2 && dir(multiFile).bytes > 0
    fidCheck = fopen(multiFile, 'r');
    oldHeader = fgetl(fidCheck);
    fclose(fidCheck);
    if isempty(strfind(oldHeader, 'WeightProfile')) || isempty(strfind(oldHeader, 'W_EnergyEff'))
        multiFile = 'single_run_weight_profiles_ACO_VNEQTS_new.csv';
    end
end
writeHeader = (exist(multiFile, 'file') ~= 2) || (dir(multiFile).bytes == 0);
fidMulti = fopen(multiFile, 'a');
if fidMulti < 0
    error('Cannot open %s for writing.', multiFile);
end
if writeHeader
    fprintf(fidMulti, '%s\n', multiHeader);
end
fprintf(fidMulti, 'ACO,%s,%d,%d,%.2f,%.2f,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d\n', ...
    weightProfileName, run_id, seed, model.W(1), model.W(2), model.W(3), model.W(4), ...
    bestMetrics.fitness, bestMetrics.phi_site, bestMetrics.phi_traf, ...
    bestMetrics.avg_utilization, bestMetrics.balance, bestMetrics.energy_eff, ...
    bestMetrics.total_energy, bestMetrics.cloud_fallback, numel(activeSites));
fclose(fidMulti);

fprintf('\n[ACO-VNEQTS final]\n');
fprintf('  Weight profile      : %s [%.2f %.2f %.2f %.2f]\n', ...
    weightProfileName, model.W(1), model.W(2), model.W(3), model.W(4));
fprintf('  Runtime             : %.2f min\n', elapsed / 60);
fprintf('  Fitness F           : %.6f\n', bestMetrics.fitness);
fprintf('  Site coverage       : %.4f%%\n', bestMetrics.phi_site * 100);
fprintf('  Traffic coverage    : %.4f%%\n', bestMetrics.phi_traf * 100);
fprintf('  Avg utilization U   : %.4f%%\n', bestMetrics.avg_utilization * 100);
fprintf('  Load balance B      : %.6f\n', bestMetrics.balance);
fprintf('  Energy efficiency Ee: %.6f\n', bestMetrics.energy_eff);
fprintf('  Total energy E      : %.6f\n', bestMetrics.total_energy);
fprintf('  Cloud fallback BS   : %d / %d\n', bestMetrics.cloud_fallback, N);
fprintf('  Active sites        : %d / %d\n', numel(activeSites), N);
fprintf('  Capacity violation  : %.6e\n', bestMetrics.capacity_violation);
fprintf('  Saved files         : %s, %s, %s\n', convPath, acuFile, allocFile);
