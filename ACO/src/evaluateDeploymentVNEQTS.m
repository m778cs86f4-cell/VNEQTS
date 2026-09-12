function metrics = evaluateDeploymentVNEQTS(n, model)
%EVALUATEDEPLOYMENTVNEQTS Fitness used by VNEQTS and comparison baselines.

    n = n(:);
    N = model.num_bs;
    if numel(n) ~= N
        error('Deployment vector length mismatch: expected %d, got %d.', N, numel(n));
    end

    active = find(n > 0);
    capacity = zeros(N, 1);
    capacity(active) = model.C0 .* n(active);

    [sigma, load_per_site] = computeServiceMappingVNEQTS(n, model, capacity);
    covered = sigma > 0;

    rho = zeros(N, 1);
    if ~isempty(active)
        rho(active) = min(load_per_site(active) ./ max(capacity(active), eps), 1.0);
    end

    phi_site = sum(covered) / N;
    if model.TOTAL_LAMBDA > 0
        phi_traf = sum(model.lambda(covered)) / model.TOTAL_LAMBDA;
    else
        phi_traf = 0.0;
    end

    rho_active = rho(active);
    if isempty(rho_active)
        avg_utilization = 0.0;
    else
        avg_utilization = mean(rho_active);
    end

    if numel(rho_active) >= 2
        mean_rho = mean(rho_active);
        std_rho = sqrt(mean((rho_active - mean_rho).^2));
        balance = 1.0 - min(1.0, std_rho / (mean_rho + model.B_EPS));
    elseif numel(rho_active) == 1
        balance = 1.0;
    else
        balance = 0.0;
    end

    total_energy = sum(n(active) .* (1.0 + rho(active)));
    energy_eff = 1.0 - (total_energy - model.M_BUDGET) / model.M_BUDGET;
    energy_eff = max(0.0, min(1.0, energy_eff));

    W = model.W;
    fitness = W(1) * phi_site + W(2) * phi_traf + W(3) * balance + W(4) * energy_eff;

    metrics = struct();
    metrics.fitness = fitness;
    metrics.phi_site = phi_site;
    metrics.phi_traf = phi_traf;
    metrics.avg_utilization = avg_utilization;
    metrics.balance = balance;
    metrics.energy_eff = energy_eff;
    metrics.total_energy = total_energy;
    metrics.cloud_fallback = N - sum(covered);
    metrics.sigma = sigma;
    metrics.load_per_site = load_per_site;
    metrics.utilization = rho;
    metrics.capacity = capacity;
    metrics.capacity_violation = max(0.0, max(load_per_site - capacity));
    metrics.active_sites = active;
end
