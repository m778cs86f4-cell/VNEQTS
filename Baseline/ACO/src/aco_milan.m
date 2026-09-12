function result = aco_milan(csvPath, seed, maxIter, nAnt, acuBudget, dMaxMeters)
%ACO_MILAN ACO baseline on the preprocessed Milan workload dataset.
%
% Usage:
%   aco_milan('milan_bs_workload.csv')
%   aco_milan('milan_bs_workload.csv', [], 500, 30, 71, 1500)

    if nargin < 1 || isempty(csvPath), csvPath = 'milan_bs_workload.csv'; end
    if nargin < 2 || isempty(seed), seed = floor(mod(now * 86400000, 2147483647)); end
    if nargin < 3 || isempty(maxIter), maxIter = 500; end
    if nargin < 4 || isempty(nAnt), nAnt = 30; end
    if nargin < 5 || isempty(acuBudget), acuBudget = 71; end
    if nargin < 6 || isempty(dMaxMeters), dMaxMeters = 1500.0; end

    if ischar(seed), seed = str2double(seed); end
    if ischar(maxIter), maxIter = str2double(maxIter); end
    if ischar(nAnt), nAnt = str2double(nAnt); end
    if ischar(acuBudget), acuBudget = str2double(acuBudget); end
    if ischar(dMaxMeters), dMaxMeters = str2double(dMaxMeters); end
    if ~isfinite(seed), seed = floor(mod(now * 86400000, 2147483647)); end
    seed = floor(mod(seed, 2147483647));
    if seed <= 0, seed = 1; end

    if exist('OCTAVE_VERSION', 'builtin') ~= 0
        rand('seed', seed);
        randn('seed', seed);
    else
        rng(seed, 'twister');
    end

    model = loadMilanAcoModel(csvPath, acuBudget, dMaxMeters);

    alpha = 1.0;
    beta = 2.0;
    rho = 0.05;
    tau0 = 1.0;
    qDeposit = 1.0;
    eliteWeight = 2.0;
    tauMin = 1e-6;
    tauMax = 100.0;

    tau = tau0 * ones(model.num_bs, 1);
    eta = model.local_workload_norm + 0.05;

    best.n = zeros(model.num_bs, 1);
    best.metrics = [];
    best.fitness = -Inf;

    convFile = 'convergence_ACO_Milan.csv';
    fidConv = fopen(convFile, 'w');
    if fidConv < 0
        error('Cannot create %s.', convFile);
    end
    fprintf(fidConv, 'iteration,best_fitness,mean_fitness,std_fitness,site_coverage,traffic_coverage,load_balance,energy_efficiency,total_energy,cloud_fallback,active_edge_sites,average_utilization\n');

    fprintf('[ACO-Milan] input=%s seed=%d ants=%d iterations=%d\n', ...
        csvPath, seed, nAnt, maxIter);
    fprintf('Conditions: M=%d, nmax=%d, W=[%.2f %.2f %.2f %.2f]\n', ...
        model.M_BUDGET, model.N_MAX, ...
        model.W(1), model.W(2), model.W(3), model.W(4));
    fprintf('Provisioned capacity: %.2f | Aggregate provisioned utilization: %.4f%%\n', ...
        model.M_BUDGET * model.acu_capacity, ...
        model.total_workload / (model.M_BUDGET * model.acu_capacity) * 100.0);

    tStart = tic;
    for it = 1:maxIter
        antSolutions = cell(nAnt, 1);
        antFitness = zeros(nAnt, 1);

        for ant = 1:nAnt
            n = constructMilanAcoSolution(tau, eta, model.M_BUDGET, ...
                model.N_MAX, alpha, beta);
            metrics = evaluateMilanAcoDeployment(n, model);

            antSolutions{ant} = n;
            antFitness(ant) = metrics.fitness;

            if metrics.fitness > best.fitness
                best.n = n;
                best.metrics = metrics;
                best.fitness = metrics.fitness;
            end
        end

        tau = (1.0 - rho) * tau;
        for ant = 1:nAnt
            tau = tau + (qDeposit * max(antFitness(ant), 0) / model.M_BUDGET) * antSolutions{ant};
        end
        tau = tau + (eliteWeight * qDeposit * max(best.fitness, 0) / model.M_BUDGET) * best.n;
        tau = min(max(tau, tauMin), tauMax);

        fprintf(fidConv, '%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%d,%d,%.8f\n', ...
            it, best.metrics.fitness, mean(antFitness), std(antFitness, 1), ...
            best.metrics.site_coverage, best.metrics.traffic_coverage, ...
            best.metrics.load_balance, best.metrics.energy_efficiency, ...
            best.metrics.total_energy, best.metrics.cloud_fallback, ...
            best.metrics.active_edge_sites, best.metrics.average_utilization);

        if it == 1 || mod(it, 50) == 0
            fprintf('Iter %d | Fitness=%.8f | Site=%.2f%% | Traffic=%.2f%% | B=%.6f | Ee=%.6f | Cloud=%d | U=%.2f%%\n', ...
                it, best.metrics.fitness, best.metrics.site_coverage * 100.0, ...
                best.metrics.traffic_coverage * 100.0, best.metrics.load_balance, ...
                best.metrics.energy_efficiency, best.metrics.cloud_fallback, ...
                best.metrics.average_utilization * 100.0);
        end
    end
    fclose(fidConv);

    elapsed = toc(tStart);
    writeMilanAcoDeploymentCsv(best.n, best.metrics, model, 'acu_deployment_ACO_Milan.csv');
    writeMilanAcoAllocationCsv(best.n, best.metrics, model, 'bs_es_allocation_ACO_Milan.csv');

    result = best.metrics;
    result.elapsed_seconds = elapsed;

    fprintf('\nFinal ACO Milan result\n');
    fprintf('Fitness: %.8f\n', result.fitness);
    fprintf('Site coverage: %.4f%%\n', result.site_coverage * 100.0);
    fprintf('Traffic coverage: %.4f%%\n', result.traffic_coverage * 100.0);
    fprintf('Load balance: %.8f\n', result.load_balance);
    fprintf('Energy efficiency: %.8f\n', result.energy_efficiency);
    fprintf('Total energy: %.8f\n', result.total_energy);
    fprintf('Cloud fallback BSs: %d / %d\n', result.cloud_fallback, model.num_bs);
    fprintf('Active edge sites: %d / %d\n', result.active_edge_sites, model.num_bs);
    fprintf('Average utilization: %.4f%%\n', result.average_utilization * 100.0);
    fprintf('Runtime: %.2f min\n', elapsed / 60.0);
    fprintf('Output: acu_deployment_ACO_Milan.csv, bs_es_allocation_ACO_Milan.csv, convergence_ACO_Milan.csv\n');
end

function model = loadMilanAcoModel(csvPath, acuBudget, dMaxMeters)
    if exist(csvPath, 'file') ~= 2
        error('Cannot find %s.', csvPath);
    end

    fid = fopen(csvPath, 'r');
    if fid < 0
        error('Cannot open %s.', csvPath);
    end
    headerLine = fgetl(fid);
    header = strsplit(strtrim(headerLine), ',');
    for h = 1:numel(header)
        header{h} = lower(strtrim(header{h}));
    end
    C = textscan(fid, '%q %f %f %f %f %f %f %f %f %f', 'Delimiter', ',');
    fclose(fid);

    idCol = findMilanHeaderIndex(header, {'bs_id', 'base_station_id', 'id'});
    latCol = findMilanHeaderIndex(header, {'latitude', 'lat'});
    lonCol = findMilanHeaderIndex(header, {'longitude', 'lon', 'lng'});
    workloadCol = findMilanHeaderIndex(header, {'lambda', 'lambda_k', 'workload'});
    if any([idCol, latCol, lonCol, workloadCol] == 0)
        error('Milan CSV must contain id, latitude, longitude, and workload columns.');
    end

    bsIds = C{idCol};
    lat = C{latCol};
    lon = C{lonCol};
    workload = C{workloadCol};

    valid = isfinite(lat) & isfinite(lon) & isfinite(workload) & workload >= 0;
    model.bs_ids = bsIds(valid);
    model.lat = lat(valid);
    model.lon = lon(valid);
    model.workload = max(workload(valid), 1e-6);
    model.num_bs = numel(model.workload);

    if ~isfinite(acuBudget) || acuBudget <= 0
        error('ACU budget must be a positive value.');
    end
    if ~isfinite(dMaxMeters) || dMaxMeters <= 0
        error('Dmax must be a positive finite distance.');
    end

    model.D_MAX = dMaxMeters;
    model.M_BUDGET = round(acuBudget);
    model.N_MAX = 3;
    % Match VNEQTS_Milan.cpp: ACU capacity is a fixed model parameter.
    model.ACU_CAPACITY = 11000.0;
    model.W = [0.25, 0.25, 0.25, 0.25];
    model.NUMERIC_EPS = 1e-10;
    model.BALANCE_EPS = 1e-9;
    model.total_workload = sum(model.workload);
    model.acu_capacity = model.ACU_CAPACITY;

    if model.M_BUDGET > model.num_bs * model.N_MAX
        error('ACU budget exceeds total per-site deployment capacity.');
    end

    fprintf('Loading Milan model: N=%d, total workload=%.8f, ACU capacity=%.8f\n', ...
        model.num_bs, model.total_workload, model.acu_capacity);
    fprintf('Computing Milan distance matrix...\n');
    model.dist_matrix_m = buildMilanAcoDistanceMatrix(model.lat, model.lon);

    localWorkload = zeros(model.num_bs, 1);
    coverageCount = zeros(model.num_bs, 1);
    for i = 1:model.num_bs
        covered = model.dist_matrix_m(i, :) <= model.D_MAX;
        localWorkload(i) = sum(model.workload(covered));
        coverageCount(i) = sum(covered);
    end
    model.local_workload = localWorkload;
    model.local_workload_norm = localWorkload ./ max(localWorkload);
    model.coverage_count_norm = coverageCount ./ max(coverageCount);
end

function idx = findMilanHeaderIndex(header, names)
    idx = 0;
    for i = 1:numel(header)
        candidate = strtrim(header{i});
        for j = 1:numel(names)
            if strcmp(candidate, names{j})
                idx = i;
                return;
            end
        end
    end
end

function D = buildMilanAcoDistanceMatrix(lat, lon)
    lat = lat(:);
    lon = lon(:);
    n = numel(lat);
    R = 6371000.0;
    latRad = lat * pi / 180.0;
    lonRad = lon * pi / 180.0;
    D = zeros(n, n, 'single');
    for i = 1:n
        dLat = latRad - latRad(i);
        dLon = lonRad - lonRad(i);
        a = sin(dLat / 2).^2 + cos(latRad(i)) .* cos(latRad) .* sin(dLon / 2).^2;
        a = max(0, min(1, a));
        c = 2 * atan2(sqrt(a), sqrt(1 - a));
        D(i, :) = single((R * c).');
    end
end

function n = constructMilanAcoSolution(tau, eta, acuBudget, maxAcusPerSite, alpha, beta)
    tau = tau(:);
    eta = eta(:);
    n = zeros(numel(tau), 1);

    for placed = 1:acuBudget
        available = n < maxAcusPerSite;
        score = (tau .^ alpha) .* (eta .^ beta);
        score(~available) = 0;
        site = rouletteMilanAco(score);
        n(site) = n(site) + 1;
    end
end

function metrics = evaluateMilanAcoDeployment(n, model)
    n = n(:);
    if numel(n) ~= model.num_bs
        error('Deployment vector length mismatch.');
    end
    if any(n < 0) || any(n > model.N_MAX) || sum(n) ~= model.M_BUDGET
        error('Infeasible deployment.');
    end

    active = find(n > 0);
    capacity = n * model.acu_capacity;
    siteLoad = zeros(model.num_bs, 1);
    sigma = zeros(model.num_bs, 1);
    coveredSites = 0;
    coveredWorkload = 0.0;

    for station = 1:model.num_bs
        w = model.workload(station);
        dists = model.dist_matrix_m(station, active);
        feasibleMask = dists <= model.D_MAX;
        feasibleSites = active(feasibleMask);
        if ~isempty(feasibleSites)
            [~, bestPos] = min(model.dist_matrix_m(station, feasibleSites));
            bestSite = feasibleSites(bestPos);
            sigma(station) = bestSite;
            siteLoad(bestSite) = siteLoad(bestSite) + w;
            coveredSites = coveredSites + 1;
            coveredWorkload = coveredWorkload + w;
        end
    end

    utilization = zeros(model.num_bs, 1);
    utilization(active) = min(siteLoad(active) ./ max(capacity(active), eps), 1.0);

    positiveSites = active(siteLoad(active) > model.NUMERIC_EPS);
    if isempty(positiveSites)
        loadBalance = 0.0;
    else
        positiveUtil = utilization(positiveSites);
        meanUtil = mean(positiveUtil);
        variance = mean((positiveUtil - meanUtil).^2);
        cv = sqrt(max(0, variance)) / (meanUtil + model.BALANCE_EPS);
        loadBalance = 1.0 - min(1.0, cv);
    end

    totalEnergy = sum(n(active) .* (1.0 + utilization(active)));
    energyEfficiency = 1.0 - (totalEnergy - model.M_BUDGET) / model.M_BUDGET;
    energyEfficiency = max(0.0, min(1.0, energyEfficiency));
    siteCoverage = coveredSites / model.num_bs;
    trafficCoverage = coveredWorkload / model.total_workload;
    if isempty(active)
        averageUtilization = 0.0;
    else
        averageUtilization = mean(utilization(active));
    end
    fitness = model.W(1) * siteCoverage + ...
        model.W(2) * trafficCoverage + ...
        model.W(3) * loadBalance + ...
        model.W(4) * energyEfficiency;

    metrics = struct();
    metrics.fitness = fitness;
    metrics.site_coverage = siteCoverage;
    metrics.traffic_coverage = trafficCoverage;
    metrics.load_balance = loadBalance;
    metrics.energy_efficiency = energyEfficiency;
    metrics.total_energy = totalEnergy;
    metrics.cloud_fallback = model.num_bs - coveredSites;
    metrics.active_edge_sites = numel(active);
    metrics.average_utilization = averageUtilization;
    metrics.utilization = utilization;
    metrics.site_load = siteLoad;
    metrics.capacity = capacity;
    metrics.sigma = sigma;
end

function idx = rouletteMilanAco(weights)
    weights = weights(:);
    weights(~isfinite(weights) | weights < 0) = 0;
    total = sum(weights);
    if total <= 0
        idx = randi(numel(weights));
        return;
    end
    c = cumsum(weights / total);
    idx = find(rand <= c, 1, 'first');
    if isempty(idx)
        idx = numel(weights);
    end
end

function writeMilanAcoDeploymentCsv(n, metrics, model, outFile)
    fid = fopen(outFile, 'w');
    if fid < 0
        error('Cannot create %s.', outFile);
    end
    fprintf(fid, 'BS_ID,Latitude,Longitude,ACU_Count,Capacity,ES_Workload,ES_Utilization_pct\n');
    active = find(n > 0);
    for p = 1:numel(active)
        i = active(p);
        fprintf(fid, '%s,%.6f,%.6f,%d,%.8f,%.8f,%.2f\n', ...
            model.bs_ids{i}, model.lat(i), model.lon(i), n(i), ...
            metrics.capacity(i), metrics.site_load(i), metrics.utilization(i) * 100.0);
    end
    fclose(fid);
end

function writeMilanAcoAllocationCsv(n, metrics, model, outFile)
    fid = fopen(outFile, 'w');
    if fid < 0
        error('Cannot create %s.', outFile);
    end
    fprintf(fid, 'BS_ID,Latitude,Longitude,Workload,Local_ACU_Count,Serving_ES_ID,Serving_ES_ACU_Count,Dist_to_ES_m\n');
    for station = 1:model.num_bs
        site = metrics.sigma(station);
        if site > 0
            servingId = model.bs_ids{site};
            servingAcus = n(site);
            distToEs = model.dist_matrix_m(station, site);
        else
            servingId = 'CLOUD';
            servingAcus = 0;
            distToEs = -1.0;
        end
        fprintf(fid, '%s,%.6f,%.6f,%.8f,%d,%s,%d,%.1f\n', ...
            model.bs_ids{station}, model.lat(station), model.lon(station), ...
            model.workload(station), n(station), servingId, servingAcus, distToEs);
    end
    fclose(fid);
end
