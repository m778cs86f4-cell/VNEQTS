# VNEQTS: Edge Server Placement Algorithms

This repository provides the source code and preprocessed experimental inputs
used for evaluating **VNEQTS** and several baseline algorithms for the
**Edge Server Placement Problem (ESPP)** in Mobile Edge Computing (MEC).

The repository currently contains implementations of the following algorithms:

- **VNEQTS** – the proposed optimization algorithm
- **ACO** – Ant Colony Optimization
- **DRLO** – DRLO baseline
- **QTS** – Quantum-inspired Tabu Search
- **nPGSAO** – nPGSAO baseline

Each algorithm is organized in an independent directory and contains its own
source code, preprocessed input data, and detailed README file.

---

## 1. Repository Structure

The repository is organized as follows:

```text
VNEQTS/
├── ACO/
│   ├── data/
│   ├── src/
│   └── README.md
│
├── DRLO/
│   ├── data/
│   ├── src/
│   └── README.md
│
├── QTS/
│   ├── data/
│   ├── src/
│   └── README.md
│
├── VNEQTS/
│   ├── data/
│   ├── src/
│   └── README.md
│
├── nPGSAO/
│   ├── data/
│   ├── src/
│   └── README.md
│
└── README.md
```

For each algorithm directory:

- `src/` contains the source code of the corresponding algorithm.
- `data/` contains the **preprocessed input data prepared for direct use by
  the corresponding implementation**.
- `README.md` provides algorithm-specific descriptions, parameter settings,
  compilation instructions, and running procedures.

---

## 2. Algorithms

### 2.1 VNEQTS

The `VNEQTS/` directory contains the implementation of the proposed VNEQTS
algorithm for solving the Edge Server Placement Problem.

VNEQTS employs an adaptive optimization framework that integrates
quantum-inspired search, population-diversity evaluation based on von Neumann
entropy, tabu mechanisms, elite preservation, feasibility repair, and local
search strategies.

Detailed implementation information, parameter settings, and running
instructions are provided in:

```text
VNEQTS/README.md
```

### 2.2 ACO

The `ACO/` directory contains the implementation of the
**Ant Colony Optimization (ACO)** algorithm used as a baseline method in the
experimental comparison.

Detailed instructions are provided in:

```text
ACO/README.md
```

### 2.3 DRLO

The `DRLO/` directory contains the implementation of the
**DRLO** baseline used in the experimental comparison.

Detailed instructions are provided in:

```text
DRLO/README.md
```

### 2.4 QTS

The `QTS/` directory contains the implementation of the
**Quantum-inspired Tabu Search (QTS)** baseline algorithm.

Detailed instructions are provided in:

```text
QTS/README.md
```

### 2.5 nPGSAO

The `nPGSAO/` directory contains the implementation of the
**nPGSAO** baseline used for comparison with VNEQTS.

Detailed instructions are provided in:

```text
nPGSAO/README.md
```

---

## 3. Datasets

The experiments in this repository are based on two real-world
telecommunication datasets:

1. **Shanghai Telecom Dataset**
2. **Telecom Italia Milan Dataset**

The original telecommunication records require preprocessing before they can
be used for the Edge Server Placement Problem.

The `data/` directory of each algorithm contains the processed experimental
inputs used by that implementation. These files have already been converted
into the corresponding algorithm input format and can therefore be used
directly when running the experiments.

---

## 4. Shanghai Telecom Dataset

The Shanghai Telecom Dataset is a real-world mobile communication dataset
collected in Shanghai, China.

The dataset contains Internet-access records from **9,481 mobile phones**
through **3,233 base stations** over a period of **six months**, with more
than **7.2 million access records**.

Each record contains information including:

- month and date;
- connection start time;
- connection end time;
- base-station location; and
- anonymized user identifier.

The base-station locations and user access records can be used to characterize
the spatial distribution of communication demand and the workloads of
different base stations.

In our experiments, invalid or inactive records are removed during
preprocessing. After preprocessing, **3,007 valid base stations** are retained
for the experimental evaluation.

The resulting information is further transformed into the site, traffic,
distance, and workload information required by the Edge Server Placement
Problem.

### Dataset Source

**Shanghai Telecom Dataset**

Source page:

https://wangshangguang.github.io/telecom_dataset/

The original dataset is provided for educational and non-commercial research
purposes. Users should follow the usage, citation, and redistribution
requirements specified by the original dataset provider.

### Processed Experimental Inputs

The files stored in the `data/` directories are the processed experimental
inputs used by the corresponding implementations.

The general processing procedure is:

```text
Shanghai Telecom raw records
        ↓
Invalid/inactive record removal
        ↓
Base-station information extraction
        ↓
Traffic/workload aggregation
        ↓
Distance and ESPP input construction
        ↓
Processed algorithm input
```

Users should consult the README file of each algorithm for the exact input
format used by that implementation.

---

## 5. Telecom Italia Milan Dataset

The second real-world dataset is obtained from the
**Telecom Italia Big Data Challenge**.

The Milan dataset contains anonymized and geographically aggregated
telecommunication activity collected in Milan, Italy.

The Milan area is represented by a regular spatial grid containing
**10,000 grid cells**, and telecommunication activities are recorded at
different time intervals.

The dataset contains several types of communication activity, including:

- Internet activity;
- incoming calls;
- outgoing calls;
- incoming SMS; and
- outgoing SMS.

For the experiments in this repository, the original Milan
telecommunication records are processed and converted into site and traffic
information suitable for the Edge Server Placement Problem.

### Dataset Sources

**Telecommunications - SMS, Call, Internet - MI**

Harvard Dataverse:

https://doi.org/10.7910/DVN/EGZHFV

**Milano Grid**

Harvard Dataverse:

https://doi.org/10.7910/DVN/QJWLFU

**Dataset description paper**

G. Barlacchi, M. De Nadai, R. Larcher, et al.,
"A multi-source dataset of urban life in the city of Milan and the Province
of Trentino," *Scientific Data*, vol. 2, Article 150055, 2015.

DOI:

https://doi.org/10.1038/sdata.2015.55

### Processed Experimental Inputs

For the ESPP experiments, the original Milan telecommunications data are
converted into the representation required by the placement algorithms.

The general processing procedure is:

```text
Telecom Italia Milan raw data
        ↓
Spatial grid/site extraction
        ↓
Telecommunication activity aggregation
        ↓
Traffic/workload construction
        ↓
ESPP input construction
        ↓
Processed algorithm input
```

The processed files used to run each implementation are stored in the
corresponding:

```text
<algorithm>/data/
```

directory.

---

## 6. Data Organization

The `data/` directory under each algorithm contains
**preprocessed experimental input files that can be directly used by the
corresponding program**.

These files are not intended to represent a complete copy of the original
raw telecommunication datasets.

The overall data workflow is:

```text
Original real-world dataset
        ↓
Data cleaning
        ↓
Traffic aggregation
        ↓
Site/base-station information extraction
        ↓
ESPP instance construction
        ↓
Algorithm-specific preprocessing
        ↓
<algorithm>/data/
```

Because the implementations of different algorithms may use different input
file structures or loading procedures, the processed data are stored
separately inside each algorithm directory.

Before running an implementation, please refer to its corresponding
`README.md` file for the exact data format and usage instructions.

---

## 7. Usage

To run one of the algorithms, enter the corresponding directory:

```text
ACO/
DRLO/
QTS/
VNEQTS/
nPGSAO/
```

Each directory contains:

```text
data/      Preprocessed experimental input data
src/       Source code
README.md  Algorithm-specific documentation
```

Please refer to the corresponding `README.md` for detailed information on:

- environment requirements;
- compilation;
- dependencies;
- parameter settings;
- input data;
- program execution; and
- output results.

For example, for VNEQTS:

```text
VNEQTS/README.md
```

---

## 8. Reproducibility

To facilitate experimental reproduction and comparison, VNEQTS and all
baseline algorithms are organized independently.

The repository provides:

- source code for each algorithm;
- preprocessed experimental inputs;
- algorithm-specific parameter descriptions; and
- independent running instructions.

The processed data in each algorithm's `data/` directory are prepared so that
the corresponding implementation can be executed directly according to its
README instructions.

---

## 9. Data Usage Notice

The original datasets remain subject to the terms and conditions specified by
their respective data providers.

In particular, users of the Shanghai Telecom Dataset should review the usage
and redistribution requirements on the original dataset webpage before using
or redistributing data derived from that source.

The inclusion of processed experimental inputs in this repository does not
replace or modify the terms associated with the original datasets.

---

## 10. Citation

If you use the code or experimental materials in this repository in your
research, please cite the corresponding paper.

The citation information for VNEQTS will be updated after publication.

```text
Citation information will be added here after publication.
```

When using the original Shanghai Telecom or Telecom Italia datasets, please
also follow the citation requirements specified by the corresponding dataset
providers.

---

## 11. Acknowledgements

We gratefully acknowledge the providers of the **Shanghai Telecom Dataset**
and the **Telecom Italia Big Data Challenge** datasets for making real-world
telecommunication data available for academic research.

These datasets provide valuable real-world traffic information for evaluating
edge server placement algorithms in Mobile Edge Computing environments.
