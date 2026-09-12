function n = constructAcuSolutionACO(tau, eta, M, nMax, alpha, beta)
%CONSTRUCTACUSOLUTIONACO Build one feasible ACU deployment vector.

    tau = tau(:);
    eta = eta(:);
    N = numel(tau);
    n = zeros(N, 1);

    for m = 1:M
        available = n < nMax;
        score = (tau .^ alpha) .* (eta .^ beta);
        score(~available) = 0;

        totalScore = sum(score);
        if totalScore <= 0 || ~isfinite(totalScore)
            candidates = find(available);
            choice = candidates(randi(numel(candidates)));
        else
            p = score ./ totalScore;
            choice = rouletteIndexACO(p);
        end
        n(choice) = n(choice) + 1;
    end
end

function idx = rouletteIndexACO(p)
    r = rand;
    c = cumsum(p);
    idx = find(r <= c, 1, 'first');
    if isempty(idx)
        idx = numel(p);
    end
end
