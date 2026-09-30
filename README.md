# Air Canvas Studio (HSV Color Tracking)

An interactive virtual drawing canvas created using Python, OpenCV, and NumPy. Draw, clear, and erase on your screen in real time using a color-tracked physical object (such as a blue marker cap or colored pointer).

---

## Features

- **Real-Time Color Tracking:** Tracks continuous motion of a target object using HSV thresholding.
- **Interactive UI Toolbar:** Built-in palette to switch drawing colors, activate an eraser, or clear the screen.
- **Sharp Compositing:** Seamlessly overlays drawing strokes onto the live webcam feed using bitwise masking.
- **On-Screen HUD:** Displays real-time FPS and active tool status.

---

## Project Structure

```text
AIR CANVAS/
├── app.py              # Core application script
├── requirements.txt    # Required dependencies
├── README.md           # Documentation
└── .gitignore          # Version control ignore rules