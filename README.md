# Event Data Analysis and Visualization

A powerful Python-based toolkit for processing, analyzing, and visualizing event-based camera data. This project provides efficient tools for event tensor generation, diffusion processing, and visualization of event data alongside RGB frames.

## 🌟 Features

- **Event Data Processing**
  - Fast event data loading and parsing
  - Efficient event tensor generation
  - Advanced diffusion processing
  - GPU-accelerated computations

- **Visualization**
  - Side-by-side video generation
  - Heatmap visualization
  - Quiver plot generation
  - Custom colormap support

- **Data Management**
  - Organized data structure
  - Efficient data loading
  - Configurable parameters
  - Progress tracking

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda package manager

## 🚀 Installation

### Option 1: Using Conda (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/event-data-analysis.git
cd event-data-analysis
```

2. Create and activate the conda environment:
```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate event_env
```

### Option 2: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/event-data-analysis.git
cd event-data-analysis
```

2. Create and activate a conda environment:
```bash
conda create -n event_env python=3.8
conda activate event_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
event-data-analysis/
├── config/
│   └── config.py           # Configuration parameters
├── data/
│   ├── __init__.py
│   ├── event_loader.py     # Event data loading utilities
│   └── frame_loader.py     # RGB frame loading utilities
├── event_processing/
│   ├── __init__.py
│   ├── event_tensor.py     # Event tensor generation
│   └── event_dataset.py    # Dataset implementation
├── processing/
│   ├── __init__.py
│   └── diffusion.py        # Diffusion processing
├── visualization/
│   ├── __init__.py
│   ├── video.py           # Video generation
│   ├── heatmap.py         # Heatmap visualization
│   └── quiver.py          # Quiver plot generation
├── main.py                # Main execution script
├── requirements.txt       # Project dependencies
├── environment.yml        # Conda environment specification
└── README.md             # Project documentation
```

## 🎯 Usage

1. Configure your parameters in `config/config.py`

2. Run the main script:
```bash
python main.py
```

The script will:
- Load event data and RGB frames
- Generate event tensors
- Apply diffusion processing
- Create visualizations
- Save results

## 📊 Output

The script generates:
- Processed event tensors
- Diffusion windows
- Side-by-side videos
- Visualization plots

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Event camera data processing community
- PyTorch team for the excellent deep learning framework
- OpenCV team for computer vision tools

## 📧 Contact

For questions and support, please open an issue in the GitHub repository. 