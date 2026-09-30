import cv2
import numpy as np
import time

class AirCanvas:
    def __init__(self, width=640, height=480):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.draw_color = (0, 255, 0)  # Default: Green
        self.brush_thickness = 5
        self.prev_point = None

    def draw(self, point, is_drawing):
        if is_drawing and point:
            if self.prev_point is None:
                self.prev_point = point
            cv2.line(self.canvas, self.prev_point, point, self.draw_color, self.brush_thickness)
            self.prev_point = point
        else:
            self.prev_point = None

    def clear(self):
        self.canvas = np.zeros_like(self.canvas)


# ==========================================
# Color Palette & Toolbar Configuration
# ==========================================
# BGR Color Definitions
PALETTE = [
    {"name": "CLEAR",   "color": (200, 200, 200), "bgr": None,          "thickness": 5},
    {"name": "ERASER",  "color": (50, 50, 50),    "bgr": (0, 0, 0),     "thickness": 30},
    {"name": "NAVY",    "color": (128, 0, 0),     "bgr": (128, 0, 0),   "thickness": 5},
    {"name": "EMERALD", "color": (80, 200, 120),  "bgr": (80, 200, 120),"thickness": 5},
    {"name": "CRIMSON", "color": (60, 20, 220),   "bgr": (60, 20, 220), "thickness": 5},
    {"name": "GOLD",    "color": (0, 215, 255),   "bgr": (0, 215, 255), "thickness": 5},
    {"name": "CORAL",   "color": (128, 128, 240), "bgr": (128, 128, 240),"thickness": 5},
    {"name": "SLATE",   "color": (112, 128, 144), "bgr": (112, 128, 144),"thickness": 5},
]

BUTTON_WIDTH = 75
BUTTON_HEIGHT = 50


def draw_toolbar(frame, active_color, is_eraser):
    """Renders top UI toolbar with professional colors and clear state indicators."""
    for i, btn in enumerate(PALETTE):
        x1 = i * BUTTON_WIDTH + 5
        y1 = 5
        x2 = x1 + BUTTON_WIDTH - 5
        y2 = y1 + BUTTON_HEIGHT

        # Highlight currently selected tool with a thick yellow border
        is_selected = (is_eraser and btn["name"] == "ERASER") or (
            not is_eraser and btn["bgr"] == active_color and btn["name"] != "ERASER"
        )
        border_color = (0, 255, 255) if is_selected else (250, 250, 250)
        border_thickness = 3 if is_selected else 1

        # Draw button box
        cv2.rectangle(frame, (x1, y1), (x2, y2), btn["color"], -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

        # Draw text label
        text_color = (255, 255, 255) if btn["name"] not in ["CLEAR", "GOLD"] else (0, 0, 0)
        cv2.putText(
            frame, btn["name"], (x1 + 4, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA
        )


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    canvas_engine = AirCanvas(640, 480)
    prev_time = time.time()

    # HSV range for tracking object (Default: Bright Blue tip/cap)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    is_eraser = False

    print("\n--- Professional Color Air-Canvas Started ---")
    print("How to use:")
    print(" 1. Move your tracked object into top boxes to select colors or Eraser.")
    print(" 2. Draw anywhere on the screen below the toolbar line.")
    print(" 3. Press 'C' to clear canvas, 'Q' to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # HSV color tracking mask
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tracking_point = None
        is_drawing = False

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 400:
                ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                tracking_point = (int(x), int(y))

                # Check if pointer is touching top toolbar
                if tracking_point[1] <= BUTTON_HEIGHT + 10:
                    btn_index = tracking_point[0] // BUTTON_WIDTH
                    if btn_index < len(PALETTE):
                        selected = PALETTE[btn_index]

                        if selected["name"] == "CLEAR":
                            canvas_engine.clear()
                        elif selected["name"] == "ERASER":
                            canvas_engine.draw_color = selected["bgr"]
                            canvas_engine.brush_thickness = selected["thickness"]
                            is_eraser = True
                        else:
                            canvas_engine.draw_color = selected["bgr"]
                            canvas_engine.brush_thickness = selected["thickness"]
                            is_eraser = False
                    
                    # Pause drawing while navigating the toolbar
                    canvas_engine.prev_point = None
                else:
                    is_drawing = True

        # Update Air Canvas
        canvas_engine.draw(tracking_point, is_drawing=is_drawing)

        # Sharp Canvas Compositing (Bitwise Masking)
        canvas_gray = cv2.cvtColor(canvas_engine.canvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        frame_bg = cv2.bitwise_and(frame, mask_inv)
        canvas_fg = cv2.bitwise_and(canvas_engine.canvas, canvas_engine.canvas)
        combined_output = cv2.add(frame_bg, canvas_fg)

        # Draw UI Toolbar & Dashboard
        draw_toolbar(combined_output, canvas_engine.draw_color, is_eraser)

        # Draw active tracker indicator dot
        if tracking_point:
            dot_color = (0, 0, 255) if not is_eraser else (255, 255, 255)
            dot_size = 10 if not is_eraser else 15
            cv2.circle(combined_output, tracking_point, dot_size, dot_color, 2)

        # HUD Status
        tool_name = "ERASER" if is_eraser else "BRUSH"
        cv2.putText(
            combined_output, f"FPS: {int(fps)} | Active Tool: {tool_name}", (20, 460),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        cv2.imshow("Air-Canvas Studio", combined_output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            canvas_engine.clear()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()