function [sigma, load_per_site] = computeServiceMappingVNEQTS(n, model, capacity)
%COMPUTESERVICEMAPPINGVNEQTS Assign each BS to a feasible active ACU site.
% sigma(k) is the serving site index, or 0 when the BS falls back to cloud.

    n = n(:);
    active = find(n > 0);
    sigma = zeros(model.num_bs, 1);
    load_per_site = zeros(model.num_bs, 1);
    if isempty(active)
        return;
    end

    hardCapacity = isfield(model, 'capacity_hard_constraint') && ...
        model.capacity_hard_constraint;

    if ~hardCapacity
        D = model.dist_matrix_m(:, active);
        [minDist, pos] = min(D, [], 2);
        covered = minDist <= model.D_MAX;
        sigma(covered) = active(pos(covered));
        if any(covered)
            load_per_site = accumarray(sigma(covered), model.lambda(covered), ...
                [model.num_bs, 1], @sum, 0);
        end
        return;
    end

    if nargin < 3 || isempty(capacity)
        capacity = zeros(model.num_bs, 1);
        capacity(active) = model.C0 .* n(active);
    end
    residual_capacity = capacity(:);

    % Large-demand BSs are assigned first so one small BS does not consume
    % the only nearby capacity needed by a larger BS.
    [~, station_order] = sort(model.lambda(:), 'descend');
    for order_idx = 1:numel(station_order)
        station = station_order(order_idx);
        demand = model.lambda(station);

        dists = model.dist_matrix_m(station, active);
        feasible_sites = active(dists <= model.D_MAX);
        if isempty(feasible_sites)
            continue;
        end

        feasible_dists = model.dist_matrix_m(station, feasible_sites);
        ranked = sortrows([double(feasible_dists(:)), double(feasible_sites(:))], [1, 2]);
        ranked_sites = ranked(:, 2);

        for site_idx = 1:numel(ranked_sites)
            site = ranked_sites(site_idx);
            if residual_capacity(site) + eps >= demand
                sigma(station) = site;
                load_per_site(site) = load_per_site(site) + demand;
                residual_capacity(site) = residual_capacity(site) - demand;
                break;
            end
        end
    end
end
