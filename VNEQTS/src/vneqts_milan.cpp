#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

constexpr int ACU_BUDGET = 71;
constexpr int MAX_ACUS_PER_SITE = 3;
constexpr double ACU_CAPACITY = 11000.0;

constexpr double W_SITE = 0.25;
constexpr double W_TRAFFIC = 0.25;
constexpr double W_BALANCE = 0.25;
constexpr double W_ENERGY = 0.25;

constexpr double THETA_MIN = 0.01 * PI;
constexpr double THETA_MAX = 0.10 * PI;
constexpr double MUTATION_MIN = 0.02;
constexpr double MUTATION_MAX = 0.20;
constexpr int TABU_MIN = 10;
constexpr int TABU_MAX = 40;

constexpr double ENTROPY_LOW = 0.30;
constexpr int STAGNATION_LIMIT = 25;
constexpr double THETA_ESCAPE = 0.18 * PI;
constexpr int TABU_ESCAPE = 15;
constexpr double SPECTRAL_ELITE_FRACTION = 0.20;
constexpr double ESCAPE_INDIVIDUAL_FRACTION = 0.30;
constexpr double ESCAPE_ACTIVE_WINDOW = 0.80;
constexpr int POPULATION_SIZE = 30;
constexpr int DEFAULT_MAX_GENERATIONS = 500;
constexpr int NUM_MEASUREMENTS = 50;
constexpr double D_MAX_METERS = 1500.0;

constexpr double NUMERIC_EPS = 1e-10;
constexpr double BALANCE_EPS = 1e-9;
constexpr double FITNESS_EPS = 1e-12;

struct BaseStation {
    string id;              
    double latitude = 0.0;  
    double longitude = 0.0;  
    double workload = 0.0;   
};

struct Individual {
    vector<int> acu_count;  
    vector<pair<double, double>> qubits;  

    double fitness = 0.0;
    double site_coverage = 0.0;
    double traffic_coverage = 0.0;
    double load_balance = 0.0;
    double energy_efficiency = 0.0;
    double total_energy = 0.0;
    int cloud_fallback = 0;
};

struct ServiceMapping {
    vector<int> serving_site;  
    vector<double> site_load;  
    int covered_sites = 0;
    double covered_workload = 0.0;
};

struct EntropyMetrics {
    double spectral = 0.0;  
};

mt19937_64 rng;
vector<vector<float>> distance_matrix;
double total_workload = 0.0;
double acu_capacity = ACU_CAPACITY;

double uniform01() {
    return generate_canonical<double, 53>(rng);
}

int uniformInt(int low, int high) {
    uniform_int_distribution<int> distribution(low, high);
    return distribution(rng);
}

string trim(string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

string lowerAscii(string value) {
    transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(tolower(c));
    });
    return value;
}

vector<string> splitCsv(const string& line) {
    vector<string> fields;
    string field;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            fields.push_back(trim(field));
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    fields.push_back(trim(field));
    return fields;
}

int findColumn(const vector<string>& header, const vector<string>& names) {
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        const string candidate = lowerAscii(trim(header[i]));
        for (const string& name : names) {
            if (candidate == name) {
                return i;
            }
        }
    }
    return -1;
}

double greatCircleDistance(double lat1, double lon1, double lat2, double lon2) {
    constexpr double EARTH_RADIUS_METERS = 6371000.0;
    const double d_lat = (lat2 - lat1) * PI / 180.0;
    const double d_lon = (lon2 - lon1) * PI / 180.0;
    const double lat1_rad = lat1 * PI / 180.0;
    const double lat2_rad = lat2 * PI / 180.0;
    const double a = sin(d_lat / 2.0) * sin(d_lat / 2.0) +
                     cos(lat1_rad) * cos(lat2_rad) *
                         sin(d_lon / 2.0) * sin(d_lon / 2.0);
    const double clamped_a = max(0.0, min(1.0, a));
    return 2.0 * EARTH_RADIUS_METERS *
           atan2(sqrt(clamped_a), sqrt(1.0 - clamped_a));
}

void buildDistanceMatrix(const vector<BaseStation>& stations) {
    const int n = static_cast<int>(stations.size());
    distance_matrix.assign(n, vector<float>(n, 0.0F));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const float distance = static_cast<float>(greatCircleDistance(
                stations[i].latitude, stations[i].longitude,
                stations[j].latitude, stations[j].longitude));
            distance_matrix[i][j] = distance;
            distance_matrix[j][i] = distance;
        }
    }
}

bool isFeasibleDeployment(const vector<int>& deployment) {
    int sum = 0;
    for (int count : deployment) {
        if (count < 0 || count > MAX_ACUS_PER_SITE) {
            return false;
        }
        sum += count;
    }
    return sum == ACU_BUDGET;
}

void validateProblemSize(int station_count) {
    if (station_count <= 0) {
        throw runtime_error("The base-station dataset is empty.");
    }
    if (ACU_BUDGET > station_count * MAX_ACUS_PER_SITE) {
        throw runtime_error(
            "ACU_BUDGET exceeds the total per-site deployment capacity.");
    }
}

void repairDeployment(vector<int>& deployment) {
    const int n = static_cast<int>(deployment.size());
    for (int& count : deployment) {
        count = max(0, min(MAX_ACUS_PER_SITE, count));
    }

    int total = accumulate(deployment.begin(), deployment.end(), 0);
    while (total > ACU_BUDGET) {
        vector<int> candidates;
        for (int i = 0; i < n; ++i) {
            if (deployment[i] > 0) {
                candidates.push_back(i);
            }
        }
        if (candidates.empty()) {
            throw runtime_error("Cannot repair an over-budget deployment.");
        }
        const int site = candidates[uniformInt(
            0, static_cast<int>(candidates.size()) - 1)];
        --deployment[site];
        --total;
    }

    while (total < ACU_BUDGET) {
        vector<int> candidates;
        for (int i = 0; i < n; ++i) {
            if (deployment[i] < MAX_ACUS_PER_SITE) {
                candidates.push_back(i);
            }
        }
        if (candidates.empty()) {
            throw runtime_error("Cannot repair an under-budget deployment.");
        }
        const int site = candidates[uniformInt(
            0, static_cast<int>(candidates.size()) - 1)];
        ++deployment[site];
        ++total;
    }
}

void repairBudget(
    vector<int>& deployment,
    const vector<pair<double, double>>& qubits) {
    const int n = static_cast<int>(deployment.size());
    for (int& count : deployment) {
        count = max(0, min(MAX_ACUS_PER_SITE, count));
    }

    int total = accumulate(deployment.begin(), deployment.end(), 0);
    if (total > ACU_BUDGET) {
        vector<pair<double, int>> priority;
        for (int i = 0; i < n; ++i) {
            if (deployment[i] > 0) {
                const double alpha2 = qubits[i].first * qubits[i].first;
                const double beta2 = qubits[i].second * qubits[i].second;
                priority.push_back({alpha2 - beta2, i});
            }
        }
        sort(priority.begin(), priority.end(), greater<pair<double, int>>());

        size_t cursor = 0;
        while (total > ACU_BUDGET) {
            const int site = priority[cursor % priority.size()].second;
            if (deployment[site] > 0) {
                --deployment[site];
                --total;
            }
            ++cursor;
        }
    } else if (total < ACU_BUDGET) {
        vector<pair<double, int>> priority;
        for (int i = 0; i < n; ++i) {
            if (deployment[i] < MAX_ACUS_PER_SITE) {
                const double alpha2 = qubits[i].first * qubits[i].first;
                const double beta2 = qubits[i].second * qubits[i].second;
                priority.push_back({beta2 - alpha2, i});
            }
        }
        sort(priority.begin(), priority.end(), greater<pair<double, int>>());

        size_t cursor = 0;
        while (total < ACU_BUDGET) {
            const int site = priority[cursor % priority.size()].second;
            if (deployment[site] < MAX_ACUS_PER_SITE) {
                ++deployment[site];
                ++total;
            }
            ++cursor;
        }
    }
}

ServiceMapping computeServiceMapping(
    const vector<int>& deployment,
    const vector<BaseStation>& stations) {
    const int n = static_cast<int>(deployment.size());
    ServiceMapping mapping;
    mapping.serving_site.assign(n, -1);
    mapping.site_load.assign(n, 0.0);

    vector<int> active_sites;
    for (int i = 0; i < n; ++i) {
        if (deployment[i] > 0) {
            active_sites.push_back(i);
        }
    }

    for (int station = 0; station < n; ++station) {
        int best_site = -1;
        float best_distance = numeric_limits<float>::infinity();

        for (int site : active_sites) {
            const float distance = distance_matrix[site][station];
            if (distance <= static_cast<float>(D_MAX_METERS) &&
                (distance < best_distance ||
                 (distance == best_distance && site < best_site))) {
                best_distance = distance;
                best_site = site;
            }
        }

        if (best_site != -1) {
            mapping.serving_site[station] = best_site;
            mapping.site_load[best_site] += stations[station].workload;
            ++mapping.covered_sites;
            mapping.covered_workload += stations[station].workload;
        }
    }
    return mapping;
}

void calculateFitness(
    Individual& individual,
    const vector<BaseStation>& stations) {
    if (!isFeasibleDeployment(individual.acu_count)) {
        throw runtime_error("Fitness was requested for an infeasible deployment.");
    }

    const int n = static_cast<int>(stations.size());
    const ServiceMapping mapping =
        computeServiceMapping(individual.acu_count, stations);

    individual.cloud_fallback = n - mapping.covered_sites;
    individual.site_coverage =
        static_cast<double>(mapping.covered_sites) / static_cast<double>(n);
    individual.traffic_coverage =
        total_workload > 0.0 ? mapping.covered_workload / total_workload : 0.0;

    vector<double> positive_load_utilizations;
    double energy = 0.0;
    for (int i = 0; i < n; ++i) {
        const int count = individual.acu_count[i];
        if (count == 0) {
            continue;
        }
        const double capacity = count * acu_capacity;
        double utilization =
            capacity > 0.0 ? mapping.site_load[i] / capacity : 0.0;
        utilization = max(0.0, min(1.0, utilization));
        if (mapping.site_load[i] > NUMERIC_EPS) {
            positive_load_utilizations.push_back(utilization);
        }
        energy += static_cast<double>(count) * (1.0 + utilization);
    }
    individual.total_energy = energy;

    if (positive_load_utilizations.empty()) {
        individual.load_balance = 0.0;
    } else {
        const double mean = accumulate(
            positive_load_utilizations.begin(),
            positive_load_utilizations.end(), 0.0) /
            static_cast<double>(positive_load_utilizations.size());
        double variance = 0.0;
        for (double utilization : positive_load_utilizations) {
            const double delta = utilization - mean;
            variance += delta * delta;
        }
        variance /= static_cast<double>(positive_load_utilizations.size());
        const double coefficient_of_variation =
            sqrt(max(0.0, variance)) / (mean + BALANCE_EPS);
        individual.load_balance =
            1.0 - min(1.0, coefficient_of_variation);
    }

    const double energy_min = static_cast<double>(ACU_BUDGET);
    const double energy_max = 2.0 * static_cast<double>(ACU_BUDGET);
    individual.energy_efficiency =
        1.0 - (energy - energy_min) / (energy_max - energy_min);
    individual.energy_efficiency =
        max(0.0, min(1.0, individual.energy_efficiency));

    individual.fitness =
        W_SITE * individual.site_coverage +
        W_TRAFFIC * individual.traffic_coverage +
        W_BALANCE * individual.load_balance +
        W_ENERGY * individual.energy_efficiency;
}

pair<double, double> uniformQubit() {
    const double amplitude = sqrt(0.5);
    return {amplitude, amplitude};
}

int measureQubit(const pair<double, double>& qubit) {
    const double probability_one = qubit.second * qubit.second;
    return uniform01() < probability_one ? 1 : 0;
}

void rotateQubit(pair<double, double>& qubit, int target_bit, double theta) {
    const double delta = target_bit == 1 ? theta : -theta;
    double angle = atan2(qubit.second, qubit.first) + delta;
    angle = max(0.0, min(PI / 2.0, angle));
    qubit.first = cos(angle);
    qubit.second = sin(angle);
}

bool deploymentIsTabu(
    const vector<vector<int>>& tabu_list,
    const vector<int>& deployment) {
    return find(tabu_list.begin(), tabu_list.end(), deployment) !=
           tabu_list.end();
}

Individual quantumLocalSearch(
    const Individual& initial,
    const vector<BaseStation>& stations,
    double theta,
    int tabu_length) {
    const int n = static_cast<int>(stations.size());
    Individual best = initial;
    vector<pair<double, double>> qubits = initial.qubits;
    vector<vector<int>> tabu_list;
    tabu_list.reserve(static_cast<size_t>(tabu_length + 1));

    for (int measurement = 0; measurement < NUM_MEASUREMENTS; ++measurement) {
        vector<int> candidate_deployment = initial.acu_count;
        for (int i = 0; i < n; ++i) {
            if (measureQubit(qubits[i]) == 1) {
                candidate_deployment[i] =
                    min(candidate_deployment[i] + 1, MAX_ACUS_PER_SITE);
            } else {
                candidate_deployment[i] = max(candidate_deployment[i] - 1, 0);
            }
        }
        repairBudget(candidate_deployment, qubits);

        Individual candidate = initial;
        candidate.acu_count = move(candidate_deployment);
        candidate.qubits = qubits;
        calculateFitness(candidate, stations);

        const bool tabu = deploymentIsTabu(tabu_list, candidate.acu_count);
        const bool aspiration =
            candidate.fitness > best.fitness + FITNESS_EPS;
        if (tabu && !aspiration) {
            continue;
        }

        tabu_list.push_back(candidate.acu_count);
        if (static_cast<int>(tabu_list.size()) > tabu_length) {
            tabu_list.erase(tabu_list.begin());
        }

        if (aspiration) {
            const vector<int> previous_best = best.acu_count;
            best = candidate;

            for (int i = 0; i < n; ++i) {
                if (best.acu_count[i] > previous_best[i]) {
                    rotateQubit(qubits[i], 1, theta);
                } else if (best.acu_count[i] < previous_best[i]) {
                    rotateQubit(qubits[i], 0, theta);
                }
            }
            best.qubits = qubits;
        }
    }
    return best;
}

void jacobiEigen(
    vector<vector<double>> matrix,
    vector<double>& eigenvalues,
    vector<vector<double>>& eigenvectors) {
    const int n = static_cast<int>(matrix.size());
    constexpr int MAX_SWEEPS = 200;
    constexpr double TOLERANCE = 1e-12;

    eigenvectors.assign(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        eigenvectors[i][i] = 1.0;
    }

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        double off_diagonal_norm2 = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                off_diagonal_norm2 += matrix[i][j] * matrix[i][j];
            }
        }
        if (sqrt(off_diagonal_norm2) < TOLERANCE) {
            break;
        }

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                if (abs(matrix[p][q]) < TOLERANCE) {
                    continue;
                }
                const double tau =
                    (matrix[q][q] - matrix[p][p]) / (2.0 * matrix[p][q]);
                const double t =
                    tau >= 0.0
                        ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                        : 1.0 / (tau - sqrt(1.0 + tau * tau));
                const double c = 1.0 / sqrt(1.0 + t * t);
                const double s = t * c;

                const double app = matrix[p][p];
                const double aqq = matrix[q][q];
                const double apq = matrix[p][q];
                matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                matrix[p][q] = 0.0;
                matrix[q][p] = 0.0;

                for (int row = 0; row < n; ++row) {
                    if (row == p || row == q) {
                        continue;
                    }
                    const double arp = matrix[row][p];
                    const double arq = matrix[row][q];
                    matrix[row][p] = c * arp - s * arq;
                    matrix[p][row] = matrix[row][p];
                    matrix[row][q] = s * arp + c * arq;
                    matrix[q][row] = matrix[row][q];
                }

                for (int row = 0; row < n; ++row) {
                    const double vrp = eigenvectors[row][p];
                    const double vrq = eigenvectors[row][q];
                    eigenvectors[row][p] = c * vrp - s * vrq;
                    eigenvectors[row][q] = s * vrp + c * vrq;
                }
            }
        }
    }

    eigenvalues.resize(n);
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = matrix[i][i];
    }
}

EntropyMetrics computeVne(
    const vector<Individual>& population,
    vector<vector<double>>& features,
    vector<vector<double>>& gram,
    vector<double>& eigenvalues,
    vector<vector<double>>& eigenvectors) {
    const int population_size = static_cast<int>(population.size());
    const int dimensions =
        population_size == 0
            ? 0
            : static_cast<int>(population.front().acu_count.size());
    if (population_size <= 1 || dimensions == 0) {
        features.clear();
        gram.clear();
        eigenvalues.clear();
        eigenvectors.clear();
        return {};
    }

    vector<double> mean(dimensions, 0.0);
    for (const Individual& individual : population) {
        for (int i = 0; i < dimensions; ++i) {
            mean[i] += individual.acu_count[i];
        }
    }
    for (double& value : mean) {
        value /= static_cast<double>(population_size);
    }

    features.assign(
        population_size, vector<double>(dimensions, 0.0));
    for (int p = 0; p < population_size; ++p) {
        double norm2 = 0.0;
        for (int i = 0; i < dimensions; ++i) {
            features[p][i] = population[p].acu_count[i] - mean[i];
            norm2 += features[p][i] * features[p][i];
        }
        const double denominator = sqrt(norm2) + NUMERIC_EPS;
        for (double& value : features[p]) {
            value /= denominator;
        }
    }

    gram.assign(
        population_size, vector<double>(population_size, 0.0));
    for (int i = 0; i < population_size; ++i) {
        for (int j = i; j < population_size; ++j) {
            const double dot = inner_product(
                features[i].begin(), features[i].end(),
                features[j].begin(), 0.0);
            gram[i][j] = dot / static_cast<double>(population_size);
            gram[j][i] = gram[i][j];
        }
    }

    jacobiEigen(gram, eigenvalues, eigenvectors);

    vector<double> positive;
    for (double eigenvalue : eigenvalues) {
        if (eigenvalue > NUMERIC_EPS) {
            positive.push_back(eigenvalue);
        }
    }
    const int rank = static_cast<int>(positive.size());
    double spectral_entropy = 0.0;
    if (rank > 1) {
        const double trace =
            accumulate(positive.begin(), positive.end(), 0.0);
        for (double eigenvalue : positive) {
            const double probability = eigenvalue / trace;
            spectral_entropy -= probability * log(probability);
        }
        spectral_entropy /= log(static_cast<double>(rank));
    }

    EntropyMetrics metrics;
    metrics.spectral = max(0.0, min(1.0, spectral_entropy));
    return metrics;
}

void spectralEscape(
    vector<Individual>& population,
    const vector<double>& eigenvalues,
    const vector<vector<double>>& eigenvectors,
    const vector<BaseStation>& stations) {
    const int population_size = static_cast<int>(population.size());
    const int dimensions = static_cast<int>(stations.size());
    if (population_size < 2 || eigenvalues.empty() ||
        static_cast<int>(eigenvectors.size()) != population_size) {
        return;
    }

    const int principal_mode = static_cast<int>(distance(
        eigenvalues.begin(),
        max_element(eigenvalues.begin(), eigenvalues.end())));
    vector<pair<double, int>> centrality;
    centrality.reserve(population_size);
    for (int p = 0; p < population_size; ++p) {
        centrality.push_back({abs(eigenvectors[p][principal_mode]), p});
    }
    sort(centrality.begin(), centrality.end(),
         [](const auto& a, const auto& b) {
             if (a.first != b.first) {
                 return a.first > b.first;
             }
             return a.second < b.second;
         });

    const int elite_count = min(
        population_size - 1,
        max(1, static_cast<int>(ceil(
            SPECTRAL_ELITE_FRACTION * population_size))));
    vector<int> elite_indices;
    vector<bool> is_elite(population_size, false);
    for (int k = 0; k < elite_count; ++k) {
        const int index = centrality[k].second;
        elite_indices.push_back(index);
        is_elite[index] = true;
    }

    vector<int> rebuild_indices;
    for (int p = 0; p < population_size; ++p) {
        if (!is_elite[p]) {
            rebuild_indices.push_back(p);
        }
    }
    shuffle(rebuild_indices.begin(), rebuild_indices.end(), rng);
    const int rebuild_count = min(
        static_cast<int>(rebuild_indices.size()),
        max(1, static_cast<int>(ceil(
            ESCAPE_INDIVIDUAL_FRACTION * rebuild_indices.size()))));
    rebuild_indices.resize(rebuild_count);

    vector<int> site_order(dimensions);
    iota(site_order.begin(), site_order.end(), 0);
    for (int individual_index : rebuild_indices) {
        shuffle(site_order.begin(), site_order.end(), rng);
        vector<int> rebuilt_deployment(dimensions, 0);
        vector<int> match_count(elite_indices.size(), 0);
        int remaining_budget = ACU_BUDGET;

        for (int site : site_order) {
            const int maximum_value =
                min(MAX_ACUS_PER_SITE, remaining_budget);
            int best_worst_match = numeric_limits<int>::max();
            vector<int> tied_values;

            for (int value = 0; value <= maximum_value; ++value) {
                int worst_match = 0;
                for (int q = 0;
                     q < static_cast<int>(elite_indices.size());
                     ++q) {
                    const int elite_index = elite_indices[q];
                    const int candidate_match = match_count[q] +
                        (population[elite_index].acu_count[site] == value);
                    worst_match = max(worst_match, candidate_match);
                }
                if (worst_match < best_worst_match) {
                    best_worst_match = worst_match;
                    tied_values.clear();
                    tied_values.push_back(value);
                } else if (worst_match == best_worst_match) {
                    tied_values.push_back(value);
                }
            }

            const int selected_value = tied_values[uniformInt(
                0, static_cast<int>(tied_values.size()) - 1)];
            rebuilt_deployment[site] = selected_value;
            remaining_budget -= selected_value;
            for (int q = 0;
                 q < static_cast<int>(elite_indices.size());
                 ++q) {
                const int elite_index = elite_indices[q];
                if (population[elite_index].acu_count[site] ==
                    selected_value) {
                    ++match_count[q];
                }
            }
        }

        Individual rebuilt = population[individual_index];
        rebuilt.acu_count = move(rebuilt_deployment);
        repairBudget(rebuilt.acu_count, rebuilt.qubits);
        calculateFitness(rebuilt, stations);
        population[individual_index] = quantumLocalSearch(
            rebuilt, stations, THETA_ESCAPE, TABU_ESCAPE);
    }
}


Individual crossover(
    const Individual& parent1,
    const Individual& parent2,
    const vector<BaseStation>& stations) {
    const int n = static_cast<int>(parent1.acu_count.size());
    Individual child = parent1;
    for (int i = 0; i < n; ++i) {
        if (uniform01() < 0.5) {
            child.acu_count[i] = parent2.acu_count[i];
            child.qubits[i] = parent2.qubits[i];
        }
    }
    repairDeployment(child.acu_count);
    calculateFitness(child, stations);
    return child;
}

void mutate(
    Individual& individual,
    const vector<BaseStation>& stations,
    double mutation_probability) {
    if (uniform01() >= mutation_probability) {
        return;
    }

    const int n = static_cast<int>(individual.acu_count.size());
    vector<int> sources;
    vector<int> destinations;
    for (int i = 0; i < n; ++i) {
        if (individual.acu_count[i] > 0) {
            sources.push_back(i);
        }
        if (individual.acu_count[i] < MAX_ACUS_PER_SITE) {
            destinations.push_back(i);
        }
    }
    if (sources.empty() || destinations.empty()) {
        return;
    }

    const int source =
        sources[uniformInt(0, static_cast<int>(sources.size()) - 1)];
    vector<int> valid_destinations;
    for (int destination : destinations) {
        if (destination != source) {
            valid_destinations.push_back(destination);
        }
    }
    if (valid_destinations.empty()) {
        return;
    }
    const int destination = valid_destinations[
        uniformInt(0, static_cast<int>(valid_destinations.size()) - 1)];

    --individual.acu_count[source];
    ++individual.acu_count[destination];
    calculateFitness(individual, stations);
}

Individual initializeIndividual(const vector<BaseStation>& stations) {
    const int n = static_cast<int>(stations.size());
    Individual individual;
    individual.acu_count.assign(n, 0);
    individual.qubits.assign(n, uniformQubit());

    int placed = 0;
    while (placed < ACU_BUDGET) {
        const int site = uniformInt(0, n - 1);
        if (individual.acu_count[site] < MAX_ACUS_PER_SITE) {
            ++individual.acu_count[site];
            ++placed;
        }
    }
    calculateFitness(individual, stations);
    return individual;
}

vector<BaseStation> readBaseStations(const string& filename) {
    ifstream input(filename);
    if (!input) {
        throw runtime_error("Cannot open input CSV: " + filename);
    }

    string line;
    if (!getline(input, line)) {
        throw runtime_error("Input CSV has no header: " + filename);
    }
    const vector<string> header = splitCsv(line);
    const int id_column =
        findColumn(header, {"bs_id", "base_station_id", "id"});
    const int latitude_column =
        findColumn(header, {"latitude", "lat"});
    const int longitude_column =
        findColumn(header, {"longitude", "lon", "lng"});
    const int workload_column =
        findColumn(header, {"lambda", "lambda_k", "workload"});

    if (id_column < 0 || latitude_column < 0 || longitude_column < 0 ||
        workload_column < 0) {
        throw runtime_error(
            "Milan CSV must contain ID, latitude, longitude, and workload columns.");
    }

    vector<BaseStation> stations;
    int line_number = 1;
    while (getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }
        const vector<string> fields = splitCsv(line);
        const int required_index = max(
            max(id_column, max(latitude_column, longitude_column)),
            workload_column);
        if (required_index >= static_cast<int>(fields.size())) {
            throw runtime_error(
                "Malformed CSV row at line " + to_string(line_number) + ".");
        }

        try {
            BaseStation station;
            station.id = fields[id_column];
            station.latitude = stod(fields[latitude_column]);
            station.longitude = stod(fields[longitude_column]);
            station.workload = stod(fields[workload_column]);
            if (!isfinite(station.latitude) || !isfinite(station.longitude) ||
                !isfinite(station.workload) || station.latitude < -90.0 ||
                station.latitude > 90.0 || station.longitude < -180.0 ||
                station.longitude > 180.0 || station.workload < 0.0) {
                throw invalid_argument("out-of-range Milan station value");
            }
            station.workload = max(station.workload, 1e-6);
            stations.push_back(move(station));
        } catch (const exception&) {
            throw runtime_error(
                "Invalid numeric value in CSV at line " +
                to_string(line_number) + ".");
        }
    }
    return stations;
}

int rouletteSelect(const vector<Individual>& population) {
    double total = 0.0;
    for (const Individual& individual : population) {
        total += max(0.0, individual.fitness);
    }
    if (total <= NUMERIC_EPS) {
        return uniformInt(0, static_cast<int>(population.size()) - 1);
    }

    const double target = uniform01() * total;
    double cumulative = 0.0;
    for (int i = 0; i < static_cast<int>(population.size()); ++i) {
        cumulative += max(0.0, population[i].fitness);
        if (cumulative >= target) {
            return i;
        }
    }
    return static_cast<int>(population.size()) - 1;
}

bool betterFitness(const Individual& a, const Individual& b) {
    return a.fitness > b.fitness;
}

void writeDeploymentCsv(
    const Individual& best,
    const vector<BaseStation>& stations,
    const ServiceMapping& mapping) {
    ofstream output("acu_deployment_VNEQTS_Milan.csv");
    if (!output) {
        throw runtime_error("Cannot create acu_deployment_VNEQTS_Milan.csv.");
    }
    output
        << "BS_ID,Latitude,Longitude,ACU_Count,Capacity,ES_Workload,"
           "ES_Utilization_pct\n";
    for (int i = 0; i < static_cast<int>(stations.size()); ++i) {
        if (best.acu_count[i] == 0) {
            continue;
        }
        const double capacity = best.acu_count[i] * acu_capacity;
        const double utilization =
            capacity > 0.0 ? mapping.site_load[i] / capacity : 0.0;
        output << stations[i].id << ','
               << fixed << setprecision(6)
               << stations[i].latitude << ','
               << stations[i].longitude << ','
               << best.acu_count[i] << ','
               << setprecision(8) << capacity << ','
               << mapping.site_load[i] << ','
               << setprecision(2) << utilization * 100.0 << '\n';
    }
}

void writeAllocationCsv(
    const Individual& best,
    const vector<BaseStation>& stations,
    const ServiceMapping& mapping) {
    ofstream output("bs_es_allocation_VNEQTS_Milan.csv");
    if (!output) {
        throw runtime_error("Cannot create bs_es_allocation_VNEQTS_Milan.csv.");
    }
    output
        << "BS_ID,Latitude,Longitude,Workload,Local_ACU_Count,"
           "Serving_ES_ID,Serving_ES_ACU_Count,Dist_to_ES_m\n";
    for (int station = 0;
         station < static_cast<int>(stations.size());
         ++station) {
        const int site = mapping.serving_site[station];
        const string serving_id = site >= 0 ? stations[site].id : "CLOUD";
        const int serving_acus = site >= 0 ? best.acu_count[site] : 0;
        const double distance =
            site >= 0 ? distance_matrix[site][station] : -1.0;
        output << stations[station].id << ','
               << fixed << setprecision(6)
               << stations[station].latitude << ','
               << stations[station].longitude << ','
               << setprecision(8) << stations[station].workload << ','
               << best.acu_count[station] << ','
               << serving_id << ','
               << serving_acus << ','
               << setprecision(1) << distance << '\n';
    }
}

double calculateAverageUtilization(
    const Individual& best,
    const ServiceMapping& mapping) {
    double utilization_sum = 0.0;
    int active_sites = 0;
    for (int i = 0; i < static_cast<int>(best.acu_count.size()); ++i) {
        const int count = best.acu_count[i];
        if (count == 0) {
            continue;
        }
        const double capacity = count * acu_capacity;
        const double utilization =
            capacity > 0.0 ? mapping.site_load[i] / capacity : 0.0;
        utilization_sum += max(0.0, min(1.0, utilization));
        ++active_sites;
    }
    return active_sites > 0
               ? utilization_sum / static_cast<double>(active_sites)
               : 0.0;
}

}  

int main(int argc, char* argv[]) {
    try {
        if (argc < 2 || argc > 4) {
            cerr << "Usage: " << argv[0]
                 << " <preprocessed_workload.csv> [random_seed] [generations]\n";
            return 1;
        }
        const string input_csv = argv[1];
        const uint64_t seed =
            argc >= 3
                ? static_cast<uint64_t>(stoull(argv[2]))
                : static_cast<uint64_t>(random_device{}());
        const int max_generations =
            argc >= 4 ? stoi(argv[3]) : DEFAULT_MAX_GENERATIONS;
        if (max_generations <= 0) {
            throw invalid_argument("The generation count must be positive.");
        }
        rng.seed(seed);

        const double weight_sum =
            W_SITE + W_TRAFFIC + W_BALANCE + W_ENERGY;
        if (abs(weight_sum - 1.0) > NUMERIC_EPS) {
            throw runtime_error("Fitness weights must sum to 1.");
        }

        vector<BaseStation> stations = readBaseStations(input_csv);
        validateProblemSize(static_cast<int>(stations.size()));

        total_workload = 0.0;
        for (const BaseStation& station : stations) {
            total_workload += station.workload;
        }
        if (total_workload <= 0.0) {
            throw runtime_error("Total workload must be positive.");
        }
        acu_capacity = ACU_CAPACITY;
        const double provisioned_capacity =
            static_cast<double>(ACU_BUDGET) * acu_capacity;
        const double aggregate_provisioned_utilization =
            total_workload / provisioned_capacity;

        cout << "Input: " << input_csv << '\n'
             << "Seed: " << seed << '\n'
             << "Generations: " << max_generations << '\n'
             << "Base stations: " << stations.size() << '\n'
             << "Total workload: " << fixed << setprecision(8)
             << total_workload << '\n'
             << "ACU budget: " << ACU_BUDGET << '\n'
             << "Building distance matrix...\n";
        buildDistanceMatrix(stations);

        vector<Individual> population;
        population.reserve(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            population.push_back(initializeIndividual(stations));
        }
        sort(population.begin(), population.end(), betterFitness);
        Individual global_best = population.front();

        ofstream convergence("convergence_VNEQTS_Milan.csv");
        if (!convergence) {
            throw runtime_error("Cannot create convergence_VNEQTS_Milan.csv.");
        }
        convergence
            << "generation,fitness,site_coverage,traffic_coverage,"
               "load_balance,energy_efficiency,total_energy,cloud_fallback,"
               "VNE,theta_over_pi,"
               "mutation_rate,tabu_length,stagnation\n";

        int stagnation = 0;
        EntropyMetrics entropy;
        for (int generation = 1;
             generation <= max_generations;
             ++generation) {
            vector<vector<double>> features;
            vector<vector<double>> gram;
            vector<vector<double>> eigenvectors;
            vector<double> eigenvalues;
            const EntropyMetrics sensing_entropy = computeVne(
                population, features, gram, eigenvalues, eigenvectors);

            const double eta =
                1.0 - static_cast<double>(generation) /
                          static_cast<double>(max_generations);
            const double drive =
                (1.0 - sensing_entropy.spectral) * eta;
            const double theta =
                THETA_MIN + (THETA_MAX - THETA_MIN) * drive;
            const double mutation_rate =
                MUTATION_MIN + (MUTATION_MAX - MUTATION_MIN) * drive;
            const int tabu_length = static_cast<int>(floor(
                TABU_MIN + (TABU_MAX - TABU_MIN) * drive));

            vector<Individual> next_generation;
            next_generation.reserve(POPULATION_SIZE);
            next_generation.push_back(population.front());

            while (static_cast<int>(next_generation.size()) <
                   POPULATION_SIZE) {
                const Individual& parent1 =
                    population[rouletteSelect(population)];
                const Individual& parent2 =
                    population[rouletteSelect(population)];
                Individual child = crossover(parent1, parent2, stations);
                mutate(child, stations, mutation_rate);
                repairDeployment(child.acu_count);
                calculateFitness(child, stations);
                child = quantumLocalSearch(
                    child, stations, theta, tabu_length);
                next_generation.push_back(move(child));
            }

            sort(
                next_generation.begin(),
                next_generation.end(),
                betterFitness);
            population = move(next_generation);

            if (population.front().fitness >
                global_best.fitness + FITNESS_EPS) {
                global_best = population.front();
                stagnation = 0;
            } else {
                ++stagnation;
            }

            entropy = computeVne(
                population, features, gram, eigenvalues, eigenvectors);

            const bool inside_escape_window =
                generation <= static_cast<int>(floor(
                    ESCAPE_ACTIVE_WINDOW * max_generations));
            if (inside_escape_window &&
                entropy.spectral < ENTROPY_LOW &&
                stagnation >= STAGNATION_LIMIT) {
                spectralEscape(
                    population, eigenvalues, eigenvectors, stations);
                sort(population.begin(), population.end(), betterFitness);
                if (population.front().fitness >
                    global_best.fitness + FITNESS_EPS) {
                    global_best = population.front();
                }
                stagnation = 0;
                entropy = computeVne(
                    population, features, gram, eigenvalues, eigenvectors);
            }

            if (generation >= 10 && generation % 10 == 0) {
                convergence
                    << generation << ','
                    << fixed << setprecision(8)
                    << global_best.fitness << ','
                    << global_best.site_coverage << ','
                    << global_best.traffic_coverage << ','
                    << global_best.load_balance << ','
                    << global_best.energy_efficiency << ','
                    << global_best.total_energy << ','
                    << global_best.cloud_fallback << ','
                    << entropy.spectral << ','
                    << theta / PI << ','
                    << mutation_rate << ','
                    << tabu_length << ','
                    << stagnation << '\n';

                cout << "generation=" << setw(4) << generation
                    << " fitness=" << fixed << setprecision(6)
                    << global_best.fitness
                    << " site=" << global_best.site_coverage
                    << " traffic=" << global_best.traffic_coverage
                    << " balance=" << global_best.load_balance
                    << " energy_eff=" << global_best.energy_efficiency
                    << '/' << stations.size()
                    << " VNE=" << entropy.spectral
                    << " stagnation=" << stagnation << '\n';
            }
        }

        const ServiceMapping final_mapping =
            computeServiceMapping(global_best.acu_count, stations);
        writeDeploymentCsv(global_best, stations, final_mapping);
        writeAllocationCsv(global_best, stations, final_mapping);

        const double average_utilization =
            calculateAverageUtilization(global_best, final_mapping);
        const int active_sites = static_cast<int>(count_if(
            global_best.acu_count.begin(),
            global_best.acu_count.end(),
            [](int count) { return count > 0; }));

        cout << "\nFinal VNEQTS result\n"
             << "Fitness: " << fixed << setprecision(8)
             << global_best.fitness << '\n'
             << "Site coverage: "
             << global_best.site_coverage * 100.0 << "%\n"
             << "Traffic coverage: "
             << global_best.traffic_coverage * 100.0 << "%\n"
             << "Load balance: " << global_best.load_balance << '\n'
             << "Energy efficiency: "
             << global_best.energy_efficiency << '\n'
             << "Active edge sites: "
             << active_sites << " / " << stations.size() << '\n'
             << "Average utilization: "
             << average_utilization * 100.0 << "%\n"
             << "Output: acu_deployment_VNEQTS_Milan.csv, "
                "bs_es_allocation_VNEQTS_Milan.csv, convergence_VNEQTS_Milan.csv\n";
        return 0;
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
