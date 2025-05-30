# Event Data Analysis and Visualization

A powerful Python-based toolkit for processing, analyzing, and visualizing event-based camera data. This project provides efficient tools for event tensor generation, diffusion processing, and visualization of event data alongside RGB frames.

![Heat Diffusion Visualization](demo/output_cropped.gif)

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

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Conda package manager

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/primal-lab/Event_Data_Analysis_Visualization.git
cd Event_Data_Analysis_Visualization
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

1. Configure your parameters in `config/config.py` Here are some special parameters:

```
EVENT_STEP → Number of events to retain per unique timestamp.

DIFFUSE_TIME → Duration (in time units) over which events are diffused; also determines the kernel's temporal depth.

MASK_RGB_FRAMES → Number of final RGB frames on which to apply masking.

GRADIENT_PLOT → If True, generates a video with RGB frames overlaid with gradient vectors, accumulated events, and diffused events. If False, the gradient overlay is omitted.

NUM_FRAMES → Total number of RGB frames to process from the selected video.

```

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

## Good for visualization
- box/seq_01
- wall/seq_00

## 📝 TODO

### High Priority
- [ ] Clean up and optimize masked event diffusion implementation
- [ ] Enhance quiver plot visualization


### Medium Priority
- [ ] Add System Arguments for the config file

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License


## 🙏 Acknowledgments

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.



