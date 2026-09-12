function model = loadBSModelVNEQTS(csvPath)
%LOADBSMODELVNEQTS Read bs_v2.cpp output and build the VNEQTS ACU model.

    fprintf('Loading BS CSV: %s\n', csvPath);
    fid = fopen(csvPath, 'r');
    if fid < 0
        error('Cannot open %s.', csvPath);
    end

    header = fgetl(fid); %#ok<NASGU>
    C = textscan(fid, '%q %f %f %f %f %f', ...
        'Delimiter', ',', 'CollectOutput', false);
    fclose(fid);

    if numel(C) < 6 || isempty(C{1})
        error('Unexpected CSV format in %s.', csvPath);
    end

    bs_ids = C{1};
    lat = C{2};
    lon = C{3};
    total_duration = C{4};
    unique_users = C{5};
    lambda_csv = C{6};

    valid = isfinite(lat) & isfinite(lon);
    bs_ids = bs_ids(valid);
    lat = lat(valid);
    lon = lon(valid);
    total_duration = total_duration(valid);
    unique_users = unique_users(valid);
    lambda_csv = lambda_csv(valid);

    T_OBS = 183.0 * 24.0 * 3600.0;
    lambda = total_duration ./ T_OBS;
    badLambda = ~isfinite(lambda) | lambda <= 0;
    lambda(badLambda) = lambda_csv(badLambda);
    lambda(~isfinite(lambda) | lambda <= 0) = 1e-6;

    model.bs_ids = bs_ids;
    model.lat = lat(:);
    model.lon = lon(:);
    model.total_duration = total_duration(:);
    model.unique_users = unique_users(:);
    model.lambda = lambda(:);
    model.w_k = model.lambda;
    model.num_bs = numel(model.lat);

    model.D_MAX = 3000.0;
    model.M_BUDGET = 71;
    model.N_MAX = 3;
    model.T_OBS = T_OBS;
    model.TOTAL_LAMBDA = sum(model.lambda);
    model.C0 = 2.0 * model.TOTAL_LAMBDA / model.M_BUDGET;
    model.W = [0.25, 0.25, 0.25, 0.25];
    model.B_EPS = 1e-9;

    fprintf('Computing Haversine distance matrix (meters)...\n');
    model.dist_matrix_m = haversineMetersMatrix(model.lat, model.lon);

    fprintf('Computing local demand heuristic within Dmax...\n');
    model.local_lambda = zeros(model.num_bs, 1);
    for i = 1:model.num_bs
        nearMask = model.dist_matrix_m(i, :).' <= model.D_MAX;
        model.local_lambda(i) = sum(model.lambda(nearMask));
    end
    model.pheremon = model.local_lambda;

    fprintf('Loaded N=%d, total lambda=%.6f, M=%d, nmax=%d\n', ...
        model.num_bs, model.TOTAL_LAMBDA, model.M_BUDGET, model.N_MAX);
end
