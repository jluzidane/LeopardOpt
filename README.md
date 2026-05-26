# LeopardOpt: A Leopard-Inspired Algorithm for Data-Efficient Multi-Objective Optimization

## Project Overview
LeopardOpt is a multi-objective optimization algorithm inspired by leopard foraging strategies, designed to break through the “data desert” challenge in material optimization—efficiently identifying optimal formulations or structures that satisfy multiple competing performance metrics under extremely severe data scarcity.
For more algorithmic details and experimental results, please refer to our Nature paper:
Lijun Liu, et al. “Breaking the data desert in multi-objective material optimization”. Submitted to Nature (May 2026).

## License & Patent Notice
- All code in this repository is open-sourced under the LeopardOpt Academic & Non-Commercial Use License (please refer to the LICENSE file in the repository root for details).
- The core LeopardOpt algorithm is protected under a pending Chinese patent application (No. 2025120494642) and corresponding overseas applications. No patent license is granted herein.
- Strictly prohibited for any form of commercial use (including but not limited to integration into commercial products, commercial services, etc.). Commercial use is strictly prohibited without written authorization. For commercial licensing, please contact: wdp@ciac.ac.cn / ljliu@ciac.ac.cn / bpeng2019@ciac.ac.cn.

## Citation
If you find our paper or this codebase helpful in your research, please cite our Nature paper:
Lijun Liu, et al. “Breaking the data desert in multi-objective material optimization”. Submitted to Nature (May 2026).
## Requirements
The algorithm is lightweight and easy to deploy, requiring only a Python runtime environment (this repository was developed and tested using Python 3.9). If a missing library is prompted during execution, simply install it as needed.
## Instructions
To ensure you can successfully reproduce the paper’s results, please note the following before running the code:
- Ensure that input files are correctly formatted so that the scripts can read and run properly.
- Ensure that all necessary configuration and input files are prepared and properly configured before running the scripts.
- The corresponding example input files and test datasets are included in their respective directories; please do not arbitrarily change their relative paths.
## Out-of-the-Box & Reproducibility
To maximize the reproducibility of our research, this repository provides the complete LeopardOpt source code.
You don’t even need to read cumbersome configuration docs: after downloading this repository, simply navigate to the demo/ folder and run python LeopardOpt.py to experience the algorithm with a single click.
After the run is completed, the program will clearly output in the terminal:
- The running time and number of iterations of the algorithm;
- The optimized formulation combinations;
- The corresponding multi-objective performance metrics of each formulation.
(Note: The leopard_demo/ folder includes test datasets of just a few kilobytes and classic test functions. There are no external massive data dependencies, and it can run through in under a minute using only basic Python libraries.)
## Contact
- Dapeng Wang: wdp@ciac.ac.cn
- Lijun Liu: ljliu@ciac.ac.cn
- Bo Peng: bpeng2019@ciac.ac.cn
