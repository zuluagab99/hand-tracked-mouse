
import cv2
import mediapipe as mp
import pyautogui
import math

# -----------------------------
# Configuration
# -----------------------------

CAMERA_INDEX = 0
PINCH_THRESHOLD = 0.045
SMOOTHING = 0.25
FRAME_MARGIN = 0.10

SCROLL_SENSITIVITY = 350
SCROLL_DEAD_ZONE = 0.002
CLICK_COOLDOWN = 0.30

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# -----------------------------
# Initialize MediaPipe
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# -----------------------------
# Initialize camera
# -----------------------------

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    hands.close()
    raise RuntimeError("Could not open webcam.")

screen_width, screen_height = pyautogui.size()

previous_x = None
previous_y = None

previous_scroll_y = None
was_pinching = False
last_click_time = 0.0

print("Hand Mouse Controller started.")
print("Index finger: move cursor")
print("Thumb + index pinch: left-click")
print("Index + middle fingers: scroll")
print("Press Q to quit.")

# -----------------------------
# Helper functions
# -----------------------------

def finger_extended(hand, tip_id, joint_id):
    """Return True when a finger is extended upward."""
    return hand.landmark[tip_id].y < (
        hand.landmark[joint_id].y - 0.015
    )


def map_to_screen(value, margin):
    """Map a normalized camera coordinate to screen space."""
    value = (value - margin) / (1 - 2 * margin)
    value = max(0.0, min(1.0, value))
    return value


# -----------------------------
# Main loop
# -----------------------------

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        # Mirror the camera for natural movement.
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        status = "NO HAND"
        height, width, _ = frame.shape

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
            )

            # Landmark positions.
            thumb = hand.landmark[4]
            index = hand.landmark[8]
            middle = hand.landmark[12]

            # Check which fingers are extended.
            index_up = finger_extended(hand, 8, 6)
            middle_up = finger_extended(hand, 12, 10)
            ring_up = finger_extended(hand, 16, 14)
            pinky_up = finger_extended(hand, 20, 18)

            # Pinch detection.
            pinch_distance = math.hypot(
                thumb.x - index.x,
                thumb.y - index.y,
            )

            is_pinching = pinch_distance < PINCH_THRESHOLD

            # Two-finger scroll mode.
            scroll_mode = (
                index_up
                and middle_up
                and not ring_up
                and not pinky_up
                and not is_pinching
            )

            if scroll_mode:
                # Track the average position of two fingertips.
                current_scroll_y = (
                    index.y + middle.y
                ) / 2

                if previous_scroll_y is not None:
                    delta_y = (
                        previous_scroll_y - current_scroll_y
                    )

                    if abs(delta_y) > SCROLL_DEAD_ZONE:
                        scroll_amount = int(
                            delta_y * SCROLL_SENSITIVITY
                        )

                        # Avoid zero-sized scroll events.
                        if scroll_amount == 0:
                            scroll_amount = (
                                1 if delta_y > 0 else -1
                            )

                        pyautogui.scroll(scroll_amount)

                previous_scroll_y = current_scroll_y
                was_pinching = False
                status = "SCROLL"

            else:
                # Reset scrolling when leaving scroll mode.
                previous_scroll_y = None

                # Convert index fingertip to screen coordinates.
                target_x = map_to_screen(
                    index.x, FRAME_MARGIN
                ) * (screen_width - 1)

                target_y = map_to_screen(
                    index.y, FRAME_MARGIN
                ) * (screen_height - 1)

                # Smooth the cursor.
                if previous_x is None:
                    previous_x = target_x
                    previous_y = target_y
                else:
                    previous_x += (
                        target_x - previous_x
                    ) * SMOOTHING

                    previous_y += (
                        target_y - previous_y
                    ) * SMOOTHING

                # Move cursor only outside scroll mode.
                pyautogui.moveTo(
                    int(previous_x),
                    int(previous_y),
                )

                # Click once when the pinch begins.
                current_time = cv2.getTickCount() / (
                    cv2.getTickFrequency()
                )

                if (
                    is_pinching
                    and not was_pinching
                    and current_time - last_click_time
                    >= CLICK_COOLDOWN
                ):
                    pyautogui.click()
                    last_click_time = current_time

                was_pinching = is_pinching
                status = (
                    "PINCH / CLICK"
                    if is_pinching
                    else "MOVE"
                )

            # Display the current mode.
            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        else:
            # Reset state when the hand disappears.
            previous_scroll_y = None
            was_pinching = False
            status = "NO HAND"

        cv2.imshow("Hand Mouse Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Hand Mouse Controller stopped.")
