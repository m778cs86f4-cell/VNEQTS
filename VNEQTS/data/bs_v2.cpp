#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <stdexcept>
#include <cstdlib>
#include <cctype>
#include <cstring>

struct BaseStation
{
    double total_duration;                      
    double lambda_k;                           
    std::unordered_set<std::string> user_set;   
    double lat;
    double lon;
};

const double SH_LAT_MIN = 30.40;
const double SH_LAT_MAX = 31.53;
const double SH_LON_MIN = 120.52;
const double SH_LON_MAX = 122.12;

const double T_OBS = 183.0 * 24.0 * 3600.0;   

std::string trim(const std::string& s)
{
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end   = s.find_last_not_of(" \t\n\r");
    std::string trimmed = (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    if (trimmed.size() >= 2 && trimmed.front() == '"' && trimmed.back() == '"')
        trimmed = trimmed.substr(1, trimmed.size() - 2);
    return trimmed;
}

bool safeStod(const std::string& s, double& out_val)
{
    try {
        std::string clean_str;
        for (char c : s)
            if (isdigit(c) || c == '.' || c == '+' || c == '-') clean_str += c;
        if (clean_str.empty()) return false;
        out_val = std::stod(clean_str);
        return true;
    } catch (...) { return false; }
}

time_t parseTime(const std::string& timeStr)
{
    struct tm tm;
    memset(&tm, 0, sizeof(tm));
    int year, month, day, hour, minute;
    char sep1, sep2, sep3;
    std::stringstream ss(timeStr);
    if (ss >> year >> sep1 >> month >> sep2 >> day >> hour >> sep3 >> minute) {
        tm.tm_year = year - 1900;
        tm.tm_mon  = month - 1;
        tm.tm_mday = day;
        tm.tm_hour = hour;
        tm.tm_min  = minute;
        tm.tm_sec  = 0;
        return mktime(&tm);
    }
    return -1;
}


void parseSingleCSV(const std::string& filename,
                    std::map<std::pair<double,double>, BaseStation>& bs_map)
{
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "error" << filename << std::endl;
        return;
    }

    std::string line;
    std::getline(infile, line);   

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> fields;
        while (std::getline(ss, segment, ','))
            fields.push_back(trim(segment));
        if (fields.size() < 7) continue;

        double lat, lon;
        if (!safeStod(fields[4], lat) || !safeStod(fields[5], lon)) continue;
        if (lat < SH_LAT_MIN || lat > SH_LAT_MAX ||
            lon < SH_LON_MIN || lon > SH_LON_MAX) continue;

        time_t t_start = parseTime(fields[2]);
        time_t t_end   = parseTime(fields[3]);
        if (t_start == -1 || t_end == -1 || t_end < t_start) continue;

        double tau_s = difftime(t_end, t_start);   
        if (tau_s <= 0) continue;

        std::string user_id = fields[6];
        if (user_id.empty()) continue;

        std::pair<double,double> loc = {lat, lon};
        BaseStation& bs = bs_map[loc];
        bs.lat = lat;
        bs.lon = lon;
        bs.total_duration += tau_s;
        bs.user_set.insert(user_id);
    }
    infile.close();
}

void calculateLambda(std::map<std::pair<double,double>, BaseStation>& bs_map)
{
    for (auto& iter : bs_map) {
        BaseStation& bs = iter.second;
        bs.lambda_k = bs.total_duration / T_OBS;
        if (bs.lambda_k < 1e-6) bs.lambda_k = 1e-6;   
    }
    std::cout << "λ_i (Little's Law,T=" << T_OBS
              << "s=" << (T_OBS / 86400.0) << "days)" << std::endl;
}

int main()
{
    std::vector<std::string> csv_files = {
        "../Dataset/data_6.1~6.15.csv",
        "../Dataset/data_6.16~6.30.csv",
        "../Dataset/data_7.1~7.15.csv",
        "../Dataset/data_7.16~7.31.csv",
        "../Dataset/data_8.1~8.15.csv",
        "../Dataset/data_8.16~8.31.csv",
        "../Dataset/data_9.1~9.15.csv",
        "../Dataset/data_9.16~9.30.csv",
        "../Dataset/data_10.1~10.15.csv",
        "../Dataset/data_10.16~10.31.csv",
        "../Dataset/data_11.1~11.15.csv",
        "../Dataset/data_11.16~11.30.csv"
    };

    std::map<std::pair<double,double>, BaseStation> bs_map;
    for (const std::string& file : csv_files)
        parseSingleCSV(file, bs_map);

    calculateLambda(bs_map);

    std::ofstream outfile("bs_statistics_all_12files.csv");
    outfile << "bs_id,latitude,longitude,total_duration(s),unique_users,lambda" << std::endl;

    int bs_counter = 1;
    for (const auto& iter : bs_map) {
        const BaseStation& bs = iter.second;
        std::string bs_id = "BS_" + std::to_string(bs_counter++);
        outfile << bs_id                                                    << ","
                << std::fixed << std::setprecision(6) << bs.lat            << ","
                << std::fixed << std::setprecision(6) << bs.lon            << ","
                << std::fixed << std::setprecision(0) << bs.total_duration << ","
                << bs.user_set.size()                                       << ","
                << std::fixed << std::setprecision(6) << bs.lambda_k       << std::endl;
    }
    outfile.close();

    double lam_sum = 0.0, lam_max = 0.0, lam_min = 1e18;
    for (const auto& iter : bs_map) {
        double lam = iter.second.lambda_k;
        lam_sum += lam;
        if (lam > lam_max) lam_max = lam;
        if (lam < lam_min) lam_min = lam;
    }
    double lam_mean = lam_sum / (double)bs_map.size();

    // std::cout << "\n=========================" << std::endl;
    // std::cout << "Total_bs:  " << bs_map.size()  << std::endl;
    // std::cout << " T： " << T_OBS << " s = " << (T_OBS / 86400.0) << " days" << std::endl;
    std::cout << "Result: bs_statistics_all_12files.csv" << std::endl;

    return 0;
}