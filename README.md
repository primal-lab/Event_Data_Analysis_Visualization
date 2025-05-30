# Event Data Analysis and Visualization

A powerful Python-based toolkit for processing, analyzing, and visualizing event-based camera data. This project provides efficient tools for event tensor generation, diffusion processing, and visualization of event data alongside RGB frames.

![Heat Diffusion Visualization](demo/output_cropped.gif)

## 🌟 Features

- **Event Data Processing**
  - Fast event data loading and parsing
  - Efficient event tensor generation
  - Advanced diffusion processing inspired by heat diffusion

- **Visualization**
  - Side-by-side video generation
  - Heatmap visualization
  - Quiver plot generation
  - Custom colormap support


## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/primal-lab/Event_Data_Analysis_Visualization.git
cd Event_Data_Analysis_Visualization
```

2. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate event_env
```

## 🎯 Usage

1. Configure your parameters in `config/config.py` Here are some special parameters:

```
EVENT_STEP → Events to keep per unique event timestamp.

DIFFUSE_TIME → Duration (in time units) over which events are diffused; also determines the kernel's temporal depth.

MASK_RGB_FRAMES → Number of final RGB frames on which to apply masking.

GRADIENT_PLOT → If True, generates a video with RGB frames overlaid with gradient vectors, accumulated events, and diffused events. If False, the gradient overlay is omitted.

NUM_FRAMES → Total number of RGB frames to process from the selected video.

POLARITY_MODE → If set to 'Positive', it will only process +1 polarity data. If set to 'Negative' it will set 0s to +1 and +1s to 0. If set to 'Both', it will convert the 0s to -1

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


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License


## 🙏 Acknowledgments

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.



