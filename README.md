# Quantitative Analytics Reference Implementation

This project provides a reference implementation of modern quantitative analytics suitable for:
- Algorithmic Differentiation (AAD)
- Longstaff-Schwartz (LS) regression 
- XVA calculations via regression and AAD

## Overview

The implementation focuses on creating a clean, modular architecture that enables:
- Path-wise sensitivity calculations
- Optimal exercise estimation for callable products
- XVA metrics computation via regression

Key features:
- Compositional approach to financial contracts
- Smooth approximations for discontinuous payoffs
- Regression-based continuation value estimation
- Framework ready for AAD integration

## Getting Started

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=matlogica/AADC-xVA-prototype)

### Recommended Reading Order

1. `observable.ipynb`: Introduces the core concepts of observables and market data abstraction
2. `contract.ipynb`: Shows how to build complex contracts from simple building blocks
3. `analytics_demo.ipynb`: Demonstrates advanced features like LS regression and XVA

### Key Concepts

The implementation is built around three main abstractions:

1. **Observables**: 
   - Represent market data and derived calculations
   - Support automatic differentiation
   - Enable path-wise sensitivities

2. **Contracts**:
   - Compositional approach to financial products
   - Support for decision points and optionality
   - Clean separation of structure and analytics

3. **Analytics**:
   - Longstaff-Schwartz implementation
   - Smooth approximations for discontinuities
   - Framework for XVA calculations

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/matlogica/AADC-xVA-prototype.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Notebooks

You can run the notebooks:
- Locally using Jupyter/Python
- In GitHub Codespaces

### Local Setup

```bash
jupyter notebook
```

Then navigate to the notebook you want to run.

## Academic References

The implementation draws from key papers in quantitative finance:
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach
- Withington, L., & Lučić, A. (2009). Noisy hedges and fuzzy payoffs: using soft computing to improve risk stability. RBC Capital Markets
- Huge, B., & Savine, A. (2020). Modern Computational Finance: AAD and Parallel Simulations
- Green, A. (2015). XVA: Credit, Funding and Capital Valuation Adjustments

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and feedback:
- Open an issue
- Submit a pull request
- Contact info@matlogica.com