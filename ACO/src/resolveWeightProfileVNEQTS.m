function [weights, profileName, profileLabel] = resolveWeightProfileVNEQTS(weightProfile, customWeights)
%RESOLVEWEIGHTPROFILEVNEQTS Return the objective weights for profile A/B/C.

    if nargin < 1 || isempty(weightProfile)
        weightProfile = 'A';
    end
    if nargin < 2
        customWeights = [];
    end

    if ~isempty(customWeights)
        weights = customWeights(:).';
        profileName = 'Custom';
        profileLabel = 'CUSTOM';
    else
        key = upper(strtrim(char(weightProfile)));
        switch key
            case {'A', 'DEFAULT', 'EQUAL', 'WEQUAL'}
                weights = [0.25, 0.25, 0.25, 0.25];
                profileName = 'ProfileA_Equal';
                profileLabel = 'A';
            case {'B', 'ENERGY', 'WENERGY'}
                weights = [0.15, 0.15, 0.20, 0.50];
                profileName = 'ProfileB_Energy';
                profileLabel = 'B';
            case {'C', 'COVERAGE', 'WCOVERAGE'}
                weights = [0.40, 0.40, 0.10, 0.10];
                profileName = 'ProfileC_Coverage';
                profileLabel = 'C';
            otherwise
                error('Unknown weight profile "%s". Use A, B, C, or pass W=[w1 w2 w3 w4].', char(weightProfile));
        end
    end

    if numel(weights) ~= 4
        error('Weight vector must contain exactly four values: [PhiSite PhiTraf Balance EnergyEff].');
    end
    if any(~isfinite(weights)) || any(weights < 0)
        error('Weights must be finite and non-negative.');
    end
    if abs(sum(weights) - 1.0) > 1e-9
        error('Weights must sum to 1. Current sum is %.12f.', sum(weights));
    end
end
