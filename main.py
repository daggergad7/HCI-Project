"""
Hands-Free Content Viewer

This application allows a user to read a document using only facial gestures.
- Gaze at the screen edges to turn pages.
- Flare nostrils to zoom in and out.
- Double-blink to scroll when zoomed in.
"""

import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance as dist
from enum import Enum
import json
import os
from pathlib import Path

class AppState(Enum):
    CALIBRATING_EYES_OPEN = 1
    CALIBRATING_EYES_CLOSED = 2
    CALIBRATING_DOUBLE_BLINK = 3
    CALIBRATING_NOSE_NEUTRAL = 4
    CALIBRATING_NOSE_FLARED = 5
    CALIBRATING_GAZE_LEFT = 6
    CALIBRATING_GAZE_RIGHT = 7
    RUNNING = 8

class ReaderController:
    def __init__(self):
        # --- Constants ---
        self.EAR_CONSEC_FRAMES = 3
        self.DOUBLE_BLINK_INTERVAL = 0.25
        self.FLARE_COOLDOWN = 2.0
        self.NOSE_FLARE_CONSEC_FRAMES = 3
        self.GAZE_DWELL_FRAMES = 15
        self.FLARE_SENSITIVITY = 0.7 # 70% of the way between neutral and flared
        self.EYE_THRESHOLD_SENSITIVITY = 0.5 # 50% of the way between open and closed
        self.CONFIG_FILE = "config.json"
        self.SCROLL_AMOUNT = 50

        # --- App State ---
        self.app_state = AppState.CALIBRATING_EYES_OPEN
        
        # --- Gesture State & Debug ---
        self.blink_counter = 0
        self.last_blink_time = 0
        self.is_blinking = False
        self.waiting_for_second_blink = False
        self.calibrating_double_blink = False
        self.calibration_first_blink_time = None
        self.double_blink_calibration_started_at = None
        self.DOUBLE_BLINK_CALIBRATION_TIMEOUT = 6.0
        self.flare_counter = 0
        self.last_flare_time = 0
        self.gaze_dwell_counter = 0
        self.current_gaze_zone = None
        self.ear = 0
        self.gaze_ratio = 0
        self.nose_flare_ratio = 0
        
        # --- Reader State ---
        self.current_page = 0
        self.zoomed = False
        self.zoom_y_offset = 0

        # --- Calibration Data ---
        self.ear_open = None
        self.ear_closed = None
        self.neutral_nose_flare_ratio = None
        self.flare_ratio_threshold = None
        self.EAR_THRESHOLD = None
        self.gaze_left_threshold = None
        self.gaze_right_threshold = None
        self.gaze_ratio_left = None
        self.gaze_ratio_right = None

        # --- dlib and camera setup ---
        print("[INFO] Loading resources...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        (self.lStart, self.lEnd) = (42, 48)
        (self.rStart, self.rEnd) = (36, 42)
        (self.nStart, self.nEnd) = (31, 36)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        # --- Load Content ---
        #self.generate_pages_if_needed()
        self.page_paths = self.load_page_paths()
        self.pages = [cv2.imread(str(path)) for path in self.page_paths]
        if not self.pages or any(p is None for p in self.pages):
            raise IOError("Could not load page images.")
        
        self.screen_h, self.screen_w = 1080, 1920

        # --- Load Calibration ---
        if self.load_calibration():
            print("[INFO] Calibration data loaded from file.")
            self.app_state = AppState.RUNNING
        else:
            print("[INFO] No calibration file found. Starting calibration.")

    def generate_pages_if_needed(self):
        """Checks if page images exist and generates them if they don't."""
        page_files = [f"page_{i}.png" for i in range(1, 5)]
        # For simplicity, we'll just regenerate them each time.
        # A check like `all(os.path.exists(f) for f in page_files)` could be used for optimization.
        
        print("[INFO] Generating page assets...")
        width, height = 1920, 1080
        bg_color = (240, 240, 240)
        font_color = (50, 50, 50)
        title_font = cv2.FONT_HERSHEY_DUPLEX
        body_font = cv2.FONT_HERSHEY_SIMPLEX

        def create_styled_page(filename, title, contents):
            page = np.ones((height, width, 3), dtype=np.uint8) * 255
            page[:] = bg_color

            # Title
            cv2.putText(page, title, (100, 150), title_font, 2, font_color, 3, cv2.LINE_AA)
            cv2.line(page, (100, 180), (width - 100, 180), font_color, 2)

            y_offset = 280
            for content_type, data in contents:
                if content_type == "text":
                    for line in data.split('\n'):
                        cv2.putText(page, line, (150, y_offset), body_font, 1.2, font_color, 2, cv2.LINE_AA)
                        y_offset += 60
                elif content_type == "space":
                    y_offset += data
                elif content_type == "diagram":
                    data(page, y_offset)
                    y_offset += 200 # Guess for diagram height

            cv2.imwrite(filename, page)
            print(f"Generated {filename}")

        # --- Page 1: Welcome ---
        create_styled_page(
            "page_1.png",
            "Welcome!",
            [
                ("text", "This is a hands-free content viewer.\n\nYou can control this application using just your eyes and facial gestures."),
                ("space", 50),
                ("text", "The next few pages will explain how it works."),
                ("space", 80),
                ("text", "To begin, please look towards the RIGHT edge of your screen to turn the page.")
            ]
        )

        # --- Page 2: Gaze Navigation ---
        def gaze_diagram(page, y_start):
            mid_x, mid_y = width // 2, y_start + 100
            arrow_length = 400
            arrow_thickness = 8
            # Arrow Right
            cv2.arrowedLine(page, (mid_x + 50, mid_y), (mid_x + 50 + arrow_length, mid_y), (0, 180, 0), arrow_thickness, tipLength=0.2)
            cv2.putText(page, "Look Right to Go Next", (mid_x + 80, mid_y - 30), body_font, 1, font_color, 2, cv2.LINE_AA)
            # Arrow Left
            cv2.arrowedLine(page, (mid_x - 50, mid_y), (mid_x - 50 - arrow_length, mid_y), (200, 50, 50), arrow_thickness, tipLength=0.2)
            cv2.putText(page, "Look Left to Go Back", (mid_x - 420, mid_y - 30), body_font, 1, font_color, 2, cv2.LINE_AA)

        create_styled_page(
            "page_2.png",
            "Page Navigation: Gaze",
            [
                ("text", "Turn pages by looking at the edges of the screen."),
                ("space", 80),
                ("diagram", gaze_diagram),
                ("space", 150),
                ("text", "Just hold your gaze for a moment to trigger the action.")
            ]
        )

        # --- Page 3: Zoom & Scroll ---
        create_styled_page(
            "page_3.png",
            "Zoom & Scroll",
            [
                ("text", "1. ZOOM: Flare your nostrils to zoom in or out."),
                ("space", 50),
                ("text", "2. SCROLL: When zoomed in, perform a quick DOUBLE BLINK to scroll down."),
                ("space", 80),
                ("text", "This allows you to read long pages without using your hands.")
            ]
        )

        # --- Page 4: Exiting ---
        create_styled_page(
            "page_4.png",
            "Settings & Exiting",
            [
                ("text", "To EXIT the application, press the 'ESC' key at any time."),
                ("space", 50),
                ("text", "To RE-CALIBRATE your gestures, press the 'r' key."),
                ("space", 80),
                ("text", "Enjoy your hands-free reading experience!")
            ]
        )

    def load_page_paths(self):
        return sorted(Path(".").glob("page_*.png"), key=self._page_sort_key)

    @staticmethod
    def _page_sort_key(path):
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return float('inf')

    def load_calibration(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.EAR_THRESHOLD = config.get("ear_threshold")
                self.flare_ratio_threshold = config.get("flare_ratio_threshold")
                self.gaze_left_threshold = config.get("gaze_left_threshold")
                self.gaze_right_threshold = config.get("gaze_right_threshold")
                double_blink_interval = config.get("double_blink_interval")
                if None not in (self.EAR_THRESHOLD, self.flare_ratio_threshold,
                                self.gaze_left_threshold, self.gaze_right_threshold,
                                double_blink_interval):
                    self.DOUBLE_BLINK_INTERVAL = double_blink_interval
                    return True
        return False

    def save_calibration(self):
        config = {
            "ear_threshold": self.EAR_THRESHOLD,
            "flare_ratio_threshold": self.flare_ratio_threshold,
            "gaze_left_threshold": self.gaze_left_threshold,
            "gaze_right_threshold": self.gaze_right_threshold,
            "double_blink_interval": self.DOUBLE_BLINK_INTERVAL
        }
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        print(f"[INFO] Calibration data saved to {self.CONFIG_FILE}")

    def start_double_blink_calibration(self, allow_outside_runtime=False):
        """Initiates calibration for the double-blink interval."""
        if self.app_state != AppState.RUNNING and not allow_outside_runtime:
            print("[WARN] Double-blink calibration is only available while the app is running.")
            return
        self.calibrating_double_blink = True
        self.calibration_first_blink_time = None
        self.double_blink_calibration_started_at = time.time()
        self.waiting_for_second_blink = False
        self.blink_counter = 0
        self.last_blink_time = 0
        self.is_blinking = False
        print("[INFO] Double-blink calibration started. Perform two quick blinks to set a new interval.")

    def handle_double_blink_calibration(self, current_time):
        """Captures consecutive blinks to derive a personalized interval."""
        if not self.calibrating_double_blink:
            return
        if self.calibration_first_blink_time is None:
            self.calibration_first_blink_time = current_time
            print("[INFO] First blink captured. Blink again at your natural double-blink speed.")
            return

        interval = max(0.05, current_time - self.calibration_first_blink_time)
        calibrated_interval = max(0.12, min(interval * 1.3, 1.5))
        self.DOUBLE_BLINK_INTERVAL = calibrated_interval
        self.calibrating_double_blink = False
        self.calibration_first_blink_time = None
        self.double_blink_calibration_started_at = None
        self.waiting_for_second_blink = False
        print(f"[INFO] Double-blink interval calibrated to {self.DOUBLE_BLINK_INTERVAL:.2f}s.")
        if self.app_state == AppState.CALIBRATING_DOUBLE_BLINK:
            self.app_state = AppState.CALIBRATING_NOSE_NEUTRAL
            print("[INFO] Double-blink calibration complete. Relax your face and press 'c' to capture your neutral nose pose.")

    def cancel_double_blink_calibration(self, reason=None):
        """Aborts calibration and resets helper state."""
        if not self.calibrating_double_blink:
            return
        self.calibrating_double_blink = False
        self.calibration_first_blink_time = None
        self.double_blink_calibration_started_at = None
        self.waiting_for_second_blink = False
        self.is_blinking = False
        if reason:
            print(reason)
        if self.app_state == AppState.CALIBRATING_DOUBLE_BLINK:
            print("[INFO] Double-blink calibration reset. Press 'c' to try again when you're ready.")

    def capture_double_blink_interval(self, gray):
        """Looks for blinks while in calibration mode and forwards events for timing."""
        if not self.calibrating_double_blink or self.EAR_THRESHOLD is None:
            return
        if self.double_blink_calibration_started_at:
            elapsed = time.time() - self.double_blink_calibration_started_at
            if elapsed > self.DOUBLE_BLINK_CALIBRATION_TIMEOUT:
                self.cancel_double_blink_calibration("[WARN] Double-blink calibration timed out. Press 'c' to try again.")
                return
        rects = self.detector(gray, 0)
        if not rects:
            return
        shape = self.predictor(gray, rects[0])
        shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])
        self.ear = (self.eye_aspect_ratio(shape_np[self.lStart:self.lEnd]) + self.eye_aspect_ratio(shape_np[self.rStart:self.rEnd])) / 2.0
        current_time = time.time()
        if self.ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
            self.is_blinking = True
        else:
            if self.blink_counter >= self.EAR_CONSEC_FRAMES:
                self.handle_blink_event(current_time, during_calibration=True)
            self.blink_counter = 0
            self.is_blinking = False

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def get_gaze_ratio(self, eye_points, gray):
        eye_region = np.array([(p.x, p.y) for p in eye_points], np.int32)
        min_x, max_x = np.min(eye_region[:, 0]), np.max(eye_region[:, 0])
        min_y, max_y = np.min(eye_region[:, 1]), np.max(eye_region[:, 1])
        eye_frame = gray[min_y:max_y, min_x:max_x]
        
        if eye_frame.size == 0: return 1.0
        eye_frame = cv2.equalizeHist(eye_frame)
        _, threshold_eye = cv2.threshold(eye_frame, 70, 255, cv2.THRESH_BINARY_INV)
        h, w = threshold_eye.shape
        left_side_white = cv2.countNonZero(threshold_eye[0:h, 0:int(w/2)])
        right_side_white = cv2.countNonZero(threshold_eye[0:h, int(w/2):w])
        return (left_side_white + 1) / (right_side_white + 1)

    def handle_blink_event(self, current_time, during_calibration=False):
        """Processes a completed blink event for calibration or runtime interactions."""
        if self.calibrating_double_blink or during_calibration:
            self.handle_double_blink_calibration(current_time)
            self.last_blink_time = current_time
            return

        if not self.zoomed:
            self.waiting_for_second_blink = False
            self.last_blink_time = current_time
            return

        if self.waiting_for_second_blink and (current_time - self.last_blink_time) <= self.DOUBLE_BLINK_INTERVAL:
            self.scroll_zoomed_content()
            self.waiting_for_second_blink = False
        else:
            self.waiting_for_second_blink = True

        self.last_blink_time = current_time

    def scroll_zoomed_content(self):
        """Scrolls the zoomed view downward by the configured amount."""
        if not self.zoomed or not self.pages:
            return
        page_img = self.pages[self.current_page]
        h = page_img.shape[0]
        zoom_h = int(h / 1.5)
        max_offset = max(0, h - zoom_h)
        self.zoom_y_offset = min(self.zoom_y_offset + self.SCROLL_AMOUNT, max_offset)

    def run(self):
        cv2.namedWindow("Content Viewer", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Content Viewer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, cam_frame = self.cap.read()
            if not ret: break
            
            cam_frame = cv2.flip(cam_frame, 1)
            gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
            
            if self.app_state != AppState.RUNNING:
                if self.app_state == AppState.CALIBRATING_DOUBLE_BLINK and self.calibrating_double_blink:
                    self.capture_double_blink_interval(gray)
                self.handle_calibration(gray, cam_frame)
            else:
                rects = self.detector(gray, 0)
                if rects:
                    shape = self.predictor(gray, rects[0])
                    self.process_gestures(shape, gray)
                    self.draw_feedback_overlay(cam_frame, shape)
                self.draw_reader_ui(cam_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("c"):
                self.handle_calibration_capture(gray)
            elif key == ord("r"):
                print("[INFO] User requested re-calibration.")
                if self.calibrating_double_blink:
                    self.cancel_double_blink_calibration("[INFO] Double-blink calibration aborted.")
                self.app_state = AppState.CALIBRATING_EYES_OPEN
        
        self.cleanup()

    def handle_calibration(self, gray, cam_frame):
        instruction_text = ""
        if self.app_state == AppState.CALIBRATING_EYES_OPEN:
            instruction_text = "First, let's calibrate your eyes. Look straight ahead with your eyes open and press 'c' to capture."
        elif self.app_state == AppState.CALIBRATING_EYES_CLOSED:
            instruction_text = "Great! Now, close your eyes and press 'c' to set the blink threshold."
        elif self.app_state == AppState.CALIBRATING_DOUBLE_BLINK:
            if self.calibrating_double_blink:
                instruction_text = "Double-blink calibration in progress. Perform two natural blinks in quick succession."
            else:
                instruction_text = "Now we'll learn your double-blink speed. Keep your head steady, press 'c' to start, then perform a natural double blink."
        elif self.app_state == AppState.CALIBRATING_NOSE_NEUTRAL:
            instruction_text = "Next, we'll calibrate for zooming. Relax your face and press 'c'."
        elif self.app_state == AppState.CALIBRATING_NOSE_FLARED:
            instruction_text = "Now, flare your nostrils as wide as you can, hold it, and press 'c'."
        elif self.app_state == AppState.CALIBRATING_GAZE_LEFT:
            instruction_text = "Finally, let's calibrate for turning pages. Look all the way to your LEFT and press 'c'."
        elif self.app_state == AppState.CALIBRATING_GAZE_RIGHT:
            instruction_text = "Perfect. Now, look all the way to your RIGHT and press 'c' to finish."
        
        # Display full-screen camera feed
        ui_screen = cv2.resize(cam_frame, (self.screen_w, self.screen_h))

        # --- Improved Text Rendering ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        font_thickness = 3
        
        # Wrap text for better display
        lines = []
        words = instruction_text.split(' ')
        line = ""
        for word in words:
            if len(line + word) < 35:
                line += word + " "
            else:
                lines.append(line)
                line = word + " "
        lines.append(line)

        total_text_height = len(lines) * 70
        
        # Add a larger, more prominent semi-transparent background
        overlay = ui_screen.copy()
        cv2.rectangle(overlay, (0, 80), (self.screen_w, 80 + total_text_height + 50), (0,0,0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, ui_screen, 1 - alpha, 0, ui_screen)

        # Draw the text lines, centered
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            text_x = (self.screen_w - text_size[0]) // 2
            text_y = 150 + i * 70
            cv2.putText(ui_screen, line, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        cv2.imshow("Content Viewer", ui_screen)

    def handle_calibration_capture(self, gray):
        rects = self.detector(gray, 0)
        if not rects:
            print("[WARN] Calibration capture failed: No face detected.")
            return

        shape = self.predictor(gray, rects[0])
        shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])

        if self.app_state == AppState.CALIBRATING_EYES_OPEN:
            self.ear_open = (self.eye_aspect_ratio(shape_np[self.lStart:self.lEnd]) + self.eye_aspect_ratio(shape_np[self.rStart:self.rEnd])) / 2.0
            self.app_state = AppState.CALIBRATING_EYES_CLOSED
            print(f"[INFO] Eyes open EAR captured: {self.ear_open:.2f}")

        elif self.app_state == AppState.CALIBRATING_EYES_CLOSED:
            self.ear_closed = (self.eye_aspect_ratio(shape_np[self.lStart:self.lEnd]) + self.eye_aspect_ratio(shape_np[self.rStart:self.rEnd])) / 2.0
            if self.ear_open is None:
                print("[WARN] Open eye EAR not calibrated. Restarting eye calibration.")
                self.app_state = AppState.CALIBRATING_EYES_OPEN
                return
            self.EAR_THRESHOLD = self.ear_closed + (self.ear_open - self.ear_closed) * self.EYE_THRESHOLD_SENSITIVITY
            self.app_state = AppState.CALIBRATING_DOUBLE_BLINK
            print(f"[INFO] Eyes closed EAR captured: {self.ear_closed:.2f}")
            print(f"[INFO] Calculated dynamic EAR threshold: {self.EAR_THRESHOLD:.2f}")

        elif self.app_state == AppState.CALIBRATING_DOUBLE_BLINK:
            if self.calibrating_double_blink:
                self.cancel_double_blink_calibration("[INFO] Restarting double-blink calibration.")
            self.start_double_blink_calibration(allow_outside_runtime=True)
            return

        elif self.app_state == AppState.CALIBRATING_NOSE_NEUTRAL:
            nostril_p1 = (shape.part(31).x, shape.part(31).y)
            nostril_p2 = (shape.part(35).x, shape.part(35).y)
            current_nostril_width = dist.euclidean(nostril_p1, nostril_p2)
            eye_p1 = (shape.part(39).x, shape.part(39).y)
            eye_p2 = (shape.part(42).x, shape.part(42).y)
            inter_ocular_distance = dist.euclidean(eye_p1, eye_p2)
            if inter_ocular_distance == 0:
                print("[WARN] Calibration failed: Inter-ocular distance is zero.")
                return
            self.neutral_nose_flare_ratio = current_nostril_width / inter_ocular_distance
            self.app_state = AppState.CALIBRATING_NOSE_FLARED
            print(f"[INFO] Neutral nose flare ratio captured: {self.neutral_nose_flare_ratio:.2f}")

        elif self.app_state == AppState.CALIBRATING_NOSE_FLARED:
            nostril_p1 = (shape.part(31).x, shape.part(31).y)
            nostril_p2 = (shape.part(35).x, shape.part(35).y)
            current_nostril_width = dist.euclidean(nostril_p1, nostril_p2)
            eye_p1 = (shape.part(39).x, shape.part(39).y)
            eye_p2 = (shape.part(42).x, shape.part(42).y)
            inter_ocular_distance = dist.euclidean(eye_p1, eye_p2)
            if inter_ocular_distance == 0:
                print("[WARN] Calibration failed: Inter-ocular distance is zero.")
                return
            max_flare_ratio = current_nostril_width / inter_ocular_distance
            if self.neutral_nose_flare_ratio is None:
                print("[WARN] Neutral ratio not calibrated yet. Restarting nose calibration.")
                self.app_state = AppState.CALIBRATING_NOSE_NEUTRAL
                return
            self.flare_ratio_threshold = self.neutral_nose_flare_ratio + \
                (max_flare_ratio - self.neutral_nose_flare_ratio) * self.FLARE_SENSITIVITY
            self.app_state = AppState.CALIBRATING_GAZE_LEFT
            print(f"[INFO] Max flare ratio captured: {max_flare_ratio:.2f}")
            print(f"[INFO] Calculated dynamic flare ratio threshold: {self.flare_ratio_threshold:.2f}")

        elif self.app_state == AppState.CALIBRATING_GAZE_LEFT:
            left_eye_points = [shape.part(i) for i in range(self.lStart, self.lEnd)]
            right_eye_points = [shape.part(i) for i in range(self.rStart, self.rEnd)]
            self.gaze_ratio_left = (self.get_gaze_ratio(left_eye_points, gray) + self.get_gaze_ratio(right_eye_points, gray)) / 2
            self.app_state = AppState.CALIBRATING_GAZE_RIGHT
            print(f"[INFO] Gaze left ratio captured: {self.gaze_ratio_left:.2f}")

        elif self.app_state == AppState.CALIBRATING_GAZE_RIGHT:
            left_eye_points = [shape.part(i) for i in range(self.lStart, self.lEnd)]
            right_eye_points = [shape.part(i) for i in range(self.rStart, self.rEnd)]
            self.gaze_ratio_right = (self.get_gaze_ratio(left_eye_points, gray) + self.get_gaze_ratio(right_eye_points, gray)) / 2
            
            if self.gaze_ratio_left is None:
                print("[WARN] Left gaze ratio not calibrated. Restarting gaze calibration.")
                self.app_state = AppState.CALIBRATING_GAZE_LEFT
                return

            # Thresholds are halfway between the calibrated left/right and a neutral gaze of 1.0
            self.gaze_left_threshold = (self.gaze_ratio_left + 1.0) / 2.0
            self.gaze_right_threshold = (self.gaze_ratio_right + 1.0) / 2.0

            self.app_state = AppState.RUNNING
            print(f"[INFO] Gaze right ratio captured: {self.gaze_ratio_right:.2f}")
            print(f"[INFO] Calculated dynamic gaze thresholds: LEFT > {self.gaze_left_threshold:.2f}, RIGHT < {self.gaze_right_threshold:.2f}")
            self.save_calibration()



    def process_gestures(self, shape, gray):
        shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])
        
        self.ear = (self.eye_aspect_ratio(shape_np[self.lStart:self.lEnd]) + self.eye_aspect_ratio(shape_np[self.rStart:self.rEnd])) / 2.0
        
        # --- Blink Detection ---
        current_time = time.time()

        if self.calibrating_double_blink and self.double_blink_calibration_started_at:
            elapsed = current_time - self.double_blink_calibration_started_at
            if elapsed > self.DOUBLE_BLINK_CALIBRATION_TIMEOUT:
                self.cancel_double_blink_calibration("[WARN] Double-blink calibration timed out. Keeping previous interval.")

        # Check if the double blink window has expired
        if self.waiting_for_second_blink and (current_time - self.last_blink_time) > self.DOUBLE_BLINK_INTERVAL:
            self.waiting_for_second_blink = False

        if self.ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
            self.is_blinking = True
        else:
            # Eye is open
            if self.blink_counter >= self.EAR_CONSEC_FRAMES:
                self.handle_blink_event(current_time)
            self.blink_counter = 0
            self.is_blinking = False
            
        # --- End of Blink Detection ---

        # Calculate nose flare ratio
        nostril_p1 = (shape.part(31).x, shape.part(31).y)
        nostril_p2 = (shape.part(35).x, shape.part(35).y)
        current_nostril_width = dist.euclidean(nostril_p1, nostril_p2)

        eye_p1 = (shape.part(39).x, shape.part(39).y) # Right eye inner corner
        eye_p2 = (shape.part(42).x, shape.part(42).y) # Left eye inner corner
        inter_ocular_distance = dist.euclidean(eye_p1, eye_p2)

        if inter_ocular_distance > 0:
            self.nose_flare_ratio = current_nostril_width / inter_ocular_distance

        # Check for nose flare gesture
        current_time = time.time()
        if self.flare_ratio_threshold and self.nose_flare_ratio > self.flare_ratio_threshold and \
           (current_time - self.last_flare_time) > self.FLARE_COOLDOWN and self.blink_counter == 0: 
            self.flare_counter += 1
        else:
            if self.flare_counter >= self.NOSE_FLARE_CONSEC_FRAMES:
                self.zoomed = not self.zoomed
                self.zoom_y_offset = 0
                self.last_flare_time = time.time()
            self.flare_counter = 0
        
        if not self.zoomed:
            left_eye_points = [shape.part(i) for i in range(self.lStart, self.lEnd)]
            right_eye_points = [shape.part(i) for i in range(self.rStart, self.rEnd)]
            self.gaze_ratio = (self.get_gaze_ratio(left_eye_points, gray) + self.get_gaze_ratio(right_eye_points, gray)) / 2
            active_zone = None
            if self.gaze_ratio > self.gaze_left_threshold: active_zone = 'left'
            elif self.gaze_ratio < self.gaze_right_threshold: active_zone = 'right'
            
            if active_zone is None:
                self.current_gaze_zone = None
                self.gaze_dwell_counter = 0
            elif active_zone == self.current_gaze_zone:
                self.gaze_dwell_counter += 1
            else: # new zone
                self.current_gaze_zone = active_zone
                self.gaze_dwell_counter = 1
            
            if self.gaze_dwell_counter > self.GAZE_DWELL_FRAMES:
                if self.current_gaze_zone == 'right': self.current_page = min(self.current_page + 1, len(self.pages) - 1)
                elif self.current_gaze_zone == 'left': self.current_page = max(self.current_page - 1, 0)
                self.gaze_dwell_counter = 0

    def draw_reader_ui(self, cam_frame):
        page_img = self.pages[self.current_page].copy()
        if self.zoomed:
            h, w, _ = page_img.shape
            zoom_h, zoom_w = int(h / 1.5), int(w / 1.5)
            max_offset = h - zoom_h
            self.zoom_y_offset = min(self.zoom_y_offset, max_offset)
            self.zoom_y_offset = max(self.zoom_y_offset, 0)
            center_x = w // 2
            crop_start_x, crop_end_x = center_x - (zoom_w // 2), center_x + (zoom_w // 2)
            page_img = page_img[self.zoom_y_offset:self.zoom_y_offset + zoom_h, crop_start_x:crop_end_x]
        
        page_img = cv2.resize(page_img, (self.screen_w, self.screen_h))
        
        # Draw the camera feed in the bottom right corner
        pip_h, pip_w = 480, 640 # Correctly increased size
        resized_cam = cv2.resize(cam_frame, (pip_w, pip_h))
        
        # Define the position for the camera feed
        pip_x_start = self.screen_w - pip_w
        pip_y_start = self.screen_h - pip_h

        # Create a border for the camera feed
        border_thickness = 3
        cv2.rectangle(page_img, 
                      (pip_x_start - border_thickness, pip_y_start - border_thickness), 
                      (self.screen_w, self.screen_h), 
                      (100, 100, 100), 
                      border_thickness)

        page_img[pip_y_start:self.screen_h, pip_x_start:self.screen_w] = resized_cam

        indicator_text = f"Page {self.current_page + 1}/{len(self.pages)}"
        cv2.putText(page_img, indicator_text, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (50, 50, 50), 2, cv2.LINE_AA)
        
        cv2.imshow("Content Viewer", page_img)

    def draw_feedback_overlay(self, cam_frame, shape):
        shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])
        cv2.drawContours(cam_frame, [cv2.convexHull(shape_np[self.lStart:self.lEnd])], -1, (0, 255, 0), 1)
        cv2.drawContours(cam_frame, [cv2.convexHull(shape_np[self.rStart:self.rEnd])], -1, (0, 255, 0), 1)
        
        y_pos, font, font_scale, font_thickness = 30, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        
        def draw_text_with_bg(frame, text, pos, font, scale, thickness, text_color, bg_color):
            text_size, _ = cv2.getTextSize(text, font, scale, thickness)
            text_w, text_h = text_size
            cv2.rectangle(frame, (pos[0], pos[1] - text_h - 5), (pos[0] + text_w, pos[1] + 5), bg_color, -1)
            cv2.putText(frame, text, pos, font, scale, text_color, thickness, cv2.LINE_AA)

        # --- Gaze Info ---
        gaze_text = f"Gaze: {self.gaze_ratio:.2f}"
        draw_text_with_bg(cam_frame, gaze_text, (10, y_pos), font, font_scale, font_thickness, (255,255,255), (0,0,0,0.5))
        y_pos += 40
        
        zone_text = f"Zone: {self.current_gaze_zone if self.current_gaze_zone else 'None'}"
        zone_color = (0, 255, 0) if self.current_gaze_zone else (255, 255, 255)
        draw_text_with_bg(cam_frame, zone_text, (10, y_pos), font, font_scale, font_thickness, zone_color, (0,0,0,0.5))
        y_pos += 40

        # --- Gaze Dwell Progress Bar ---
        if self.current_gaze_zone:
            progress = self.gaze_dwell_counter / self.GAZE_DWELL_FRAMES
            bar_w = int(progress * 150)
            cv2.rectangle(cam_frame, (10, y_pos - 10), (160, y_pos + 10), (255, 255, 255), 1)
            cv2.rectangle(cam_frame, (10, y_pos - 10), (10 + bar_w, y_pos + 10), (0, 255, 0), -1)
        y_pos += 30

        # --- Blink Info ---
        ear_text = f"EAR: {self.ear:.2f} (T: {self.EAR_THRESHOLD:.2f})"
        draw_text_with_bg(cam_frame, ear_text, (10, y_pos), font, font_scale, font_thickness, (255,255,255), (0,0,0,0.5))
        y_pos += 40
        
        blink_text = "BLINK!" if self.blink_counter > 0 else ""
        if blink_text:
            draw_text_with_bg(cam_frame, blink_text, (10, y_pos), font, font_scale, font_thickness, (0, 255, 255), (0,0,0,0.5))
        y_pos += 40

        # --- Nose Flare Info ---
        nose_text = f"Nose: {self.nose_flare_ratio:.2f} (T: {self.flare_ratio_threshold:.2f})"
        draw_text_with_bg(cam_frame, nose_text, (10, y_pos), font, font_scale, font_thickness, (255,255,255), (0,0,0,0.5))
        y_pos += 40
        
        flare_status = "FLARING" if self.flare_counter > 0 else "NEUTRAL"
        flare_text = f"Flare: {flare_status}"
        flare_color = (0, 165, 255) if self.flare_counter > 0 else (255, 255, 255)
        draw_text_with_bg(cam_frame, flare_text, (10, y_pos), font, font_scale, font_thickness, flare_color, (0,0,0,0.5))


    def cleanup(self):
        print("[INFO] Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        controller = ReaderController()
        controller.run()
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()
