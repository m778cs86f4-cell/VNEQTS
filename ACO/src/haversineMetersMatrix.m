function D = haversineMetersMatrix(lat_deg, lon_deg)
% N×N Haversine 距离矩阵（米），与 C++ QTSHEA 一致。
    lat_deg = lat_deg(:);
    lon_deg = lon_deg(:);
    n = numel(lat_deg);
    R = 6371000.0;
    lat1 = lat_deg * pi / 180.0;
    lon1 = lon_deg * pi / 180.0;
    D = zeros(n, n);
    for i = 1:n
        dlat = (lat_deg - lat_deg(i)) * pi / 180.0;
        dlon = (lon_deg - lon_deg(i)) * pi / 180.0;
        a = sin(dlat/2).^2 + cos(lat1(i)) * cos(lat1) .* sin(dlon/2).^2;
        c = 2 * atan2(sqrt(a), sqrt(1 - a));
        D(i, :) = (R * c).';
    end
end
