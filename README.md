# CENOP

<img src="static/CENOP_logo.png" alt="CENOP Logo" height="80">

**CETacean Noise-Population Model**

CENOP is a Python translation of the DEPONS (Disturbance Effects of POrpoises in the North Sea) agent-based model. It simulates how harbour porpoise population dynamics are affected by disturbances from offshore wind farm construction and ship noise.

## Features

- 🐬 Agent-based simulation of harbour porpoise populations
- 🗺️ Realistic North Sea landscape with bathymetry and food distribution
- 🌊 Realistic Central Baltic landscape with bathymetry and food distribution
- 🔊 Noise disturbance modeling (pile-driving and ship noise)
- 📊 Interactive Shiny web interface
- 📈 Real-time visualization of population dynamics

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/cenop.git
cd cenop

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

```bash
# Run the Shiny application
shiny run app.py
```

Then open your browser to http://localhost:8000

## Project Structure

```
cenop/
├── app.py                  # Shiny application entry point
├── src/cenop/              # Core simulation package
│   ├── core/               # Simulation engine
│   ├── agents/             # Agent definitions
│   ├── behavior/           # Behavioral modules
│   ├── landscape/          # Environmental data
│   └── parameters/         # Configuration
├── ui/                     # Shiny UI components
├── data/                   # Landscape data files
└── tests/                  # Test suite
```

## License

This project is licensed under the GNU General Public License v2.0, following the original DEPONS model.

## Acknowledgments

- Original DEPONS model by Jacob Nabe-Nielsen, Aarhus University
- EU Horizon 2020 SATURN project (GA 101006443)
- AI4WIND project team
