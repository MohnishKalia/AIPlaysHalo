import signal
import subprocess
import time
import os
import mss
import numpy as np
import vgamepad as vg
from typing import Optional, Tuple, List, Any, Dict
import numpy.typing as npt
import win32gui # type: ignore
import win32con # type: ignore
import win32api # type: ignore
import random
from pathlib import Path

# import torch # Import torch when model loading is implemented

assert os.name == 'nt', "This script is designed for Windows only."

# --- Configuration ---
HALO_CE_DIR = r"C:\Program Files (x86)\Microsoft Games\Halo Custom Edition" # Adjust if necessary
HALOCEDED_EXE = os.path.join(HALO_CE_DIR, "haloceded.exe")
MULTICLIENT_EXE = os.path.join(HALO_CE_DIR, "multiclient.exe") # Assumes multiclient.exe is in HALO_CE_DIR
HALOCE_EXE = os.path.join(HALO_CE_DIR, "haloce.exe")
INIT_TXT_PATH = os.path.join(HALO_CE_DIR, r"mods\init.txt") # Assumes init.txt is in mods/HALO_CE_DIR

NUM_CLIENTS = 4 # Number of clients to launch (for now, just one)
assert NUM_CLIENTS <= 8, "NUM_CLIENTS must be 1-8 for tiling to work properly."

# WIN_WIDTH = 480
# WIN_HEIGHT = 360
WIN_WIDTH = 640
WIN_HEIGHT = 480
# WIN_WIDTH = 320
# WIN_HEIGHT = 240
TARGET_FPS = 30 # Target FPS for the game window
CAPTURE_FPS = 15 # Target FPS for capture/inference loop

SERVER_IP = "127.0.0.1"
SERVER_PORT = 2302

# --- Process Management ---
def start_dedicated_server(halo_dir: str, init_path: str) -> Optional[subprocess.Popen]:
    """Launches the Halo CE dedicated server."""
    print(f"Starting dedicated server using {init_path}...")
    cmd = [HALOCEDED_EXE, "-exec", init_path]
    try:
        print(f"Command: {' '.join(cmd)}")
        # Use DETACHED_PROCESS flag on Windows to run independently
        # Or simply run in background without waiting
        server_process = subprocess.Popen(cmd, cwd=halo_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
        print(f"Dedicated server process started (PID: {server_process.pid}).")
        return server_process
    except FileNotFoundError:
        print(f"Error: {HALOCEDED_EXE} not found. Is HALO_CE_DIR correct?")
        return None
    except Exception as e:
        print(f"Error starting dedicated server: {e}")
        return None

def start_halo_client(halo_dir: str, multiclient_path: str, haloce_path: str, 
                     width: int, height: int, fps: int, ip: str, port: int) -> Optional[subprocess.Popen]:
    """Launches a Halo CE client using multiclient.exe."""
    print(f"Starting Halo CE client ({width}x{height}@{fps}fps) connecting to {ip}:{port}...")
    cmd = [
        multiclient_path,
        r'.\haloce.exe',
        "-console",
        "-window",
        "-nosound",
        "-novideo",
        f"-vidmode", f"{width},{height},{fps}",
        "-connect", f"{ip}:{port}"
    ]
    try:
        print(f"Command: {' '.join(cmd)}")
        # Use DETACHED_PROCESS flag on Windows to run independently
        client_process = subprocess.Popen(cmd, cwd=halo_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
        print(f"Halo client process started (PID: {client_process.pid}).")
        return client_process
    except FileNotFoundError:
        print(f"Error: {multiclient_path} or {haloce_path} not found. Is HALO_CE_DIR correct?")
        return None
    except Exception as e:
        print(f"Error starting Halo client: {e}")
        return None

# --- Screen Capture ---
def setup_capture(width: int, height: int) -> Tuple[mss.mss, Dict[str, int]]:
    """Sets up the screen capture region."""
    # TODO: Find a way to target the specific Halo window instead of a fixed region
    # For now, assumes window is at top-left (0,0) or requires manual placement.
    monitor = {"top": 0, "left": 0, "width": width, "height": height}
    sct = mss.mss()
    print(f"Screen capture set up for region: {monitor}")
    return sct, monitor

def setup_client_captures(hwnds: list) -> list:
    """Sets up capture regions for all client windows based on their HWNDs."""
    import mss
    sct = mss.mss()
    monitors = []
    for hwnd in hwnds:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        monitor = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        monitors.append(monitor)
    return sct, monitors

# TODO: get better mss type
def capture_frame(sct: mss.mss, monitor: Dict[str, int]) -> npt.NDArray:
    """Captures a frame from the specified monitor region."""
    img_bgra = sct.grab(monitor)
    img_rgb = np.array(img_bgra)[:, :, :3] # Convert BGRA to RGB
    return img_rgb

# --- AI Inference (Placeholders) ---
def load_model() -> Any:
    """Loads the YOLOv5 model."""
    print("Loading YOLOv5 model... (Placeholder)")
    # Example: model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # model.fp16() # If using FP16
    # model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # print(f"Model loaded on {device}. (Placeholder)")
    model = None # Placeholder
    return model

def run_inference(model: Any, frame_rgb: npt.NDArray) -> List[Any]:
    """Runs inference on the captured frame."""
    # print("Running inference... (Placeholder)")
    # Preprocess frame (resize, tensor, normalize)
    # results = model(processed_frame)
    # Process results (get boxes, classes, confidences)
    detections = [] # Placeholder
    return detections

# --- Control Policy & Input (Placeholders) ---
def setup_gamepad(init_left_stick: tuple = None) -> Optional[vg.VX360Gamepad]:
    """Initializes and returns a virtual gamepad. Optionally sets initial left stick position."""
    try:
        pad = vg.VX360Gamepad()
        if init_left_stick is not None:
            x, y = init_left_stick
            pad.left_joystick_float(x_value_float=x, y_value_float=y)
            pad.update()
        return pad
    except Exception as e:
        print(f"Error initializing gamepad: {e}")
        print("Ensure ViGEmBus driver is installed: https://github.com/ViGEm/ViGEmBus/releases")
        return None

def apply_policy_and_control(detections: List[Any], pad: vg.VX360Gamepad) -> None:
    """Applies a simple control policy based on detections."""
    # print("Applying control policy... (Placeholder)")
    # --- Simple Placeholder Policy ---
    # If enemy detected (e.g., class 'person' if using COCO-trained YOLO):
    #   Aim towards the center of the first detected enemy bbox
    #   Press trigger
    # Else:
    #   Move forward slightly
    #   Reset aim/trigger
    # ---

    # Example (needs actual detection data):
    enemy_detected = False # Replace with actual check
    if enemy_detected:
        # target_x, target_y = get_enemy_center(detections[0]) # Placeholder
        # aim_x = (target_x / WIN_WIDTH) * 2 - 1 # Normalize to -1 to 1
        # aim_y = (target_y / WIN_HEIGHT) * 2 - 1 # Normalize to -1 to 1
        # pad.right_joystick_float(x_value_float=aim_x, y_value_float=-aim_y) # Y is often inverted
        # pad.right_trigger_float(value_float=1.0)
        # pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0) # Stop moving while aiming/shooting
        pass # Replace with actual logic
    else:
        pad.left_joystick_float(x_value_float=0.0, y_value_float=0.5) # Move forward slowly
        pad.right_joystick_float(x_value_float=0.0, y_value_float=0.0) # Reset aim
        pad.right_trigger_float(value_float=0.0) # Release trigger

    try:
        pad.update()
    except Exception as e:
        print(f"Error updating gamepad: {e}")

# --- Window Management ---
def find_halo_windows(max_clients: int = 8) -> list:
    """Finds window handles for Halo CE clients (haloce.exe)."""
    def enum_handler(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            if "Halo" == title or class_name == "Halo":
                hwnds.append(hwnd)
    hwnds = []
    win32gui.EnumWindows(enum_handler, hwnds)
    # Filter for unique handles, limit to max_clients
    return hwnds[:max_clients]

def tile_windows(hwnds: list, width: int, height: int, screen_width: int = 1920, screen_height: int = 1080):
    """Tiles the given window handles in a 2-row, 4-column grid centered on the screen, filling by column, and brings them to the front."""
    n = len(hwnds)
    cols = 4
    rows = 2
    tile_w = min(width, screen_width // cols)
    tile_h = min(height, screen_height // rows)
    grid_w = tile_w * cols
    grid_h = tile_h * rows
    offset_x = (screen_width - grid_w) // 2
    offset_y = (screen_height - grid_h) // 2
    for idx, hwnd in enumerate(hwnds):
        col = idx // rows
        row = idx % rows
        x = offset_x + col * tile_w
        y = offset_y + row * tile_h
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, win32con.WS_VISIBLE | win32con.WS_POPUP)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, x, y, tile_w, tile_h, win32con.SWP_SHOWWINDOW)
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, x, y, tile_w, tile_h, win32con.SWP_SHOWWINDOW)
        win32gui.BringWindowToTop(hwnd)

def send_dedicated_server_to_back():
    """Finds the HaloConsole window and sends it to the back."""
    def enum_handler(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "Halo Console" in title:
                result.append(hwnd)
    result = []
    win32gui.EnumWindows(enum_handler, result)
    for hwnd in result:
        win32gui.SetWindowPos(hwnd, win32con.HWND_BOTTOM, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

# --- Main Execution ---
if __name__ == "__main__":
    server_proc = None
    client_procs = [None] * NUM_CLIENTS # Preallocate list for client spawn processes
    client_pids = [-1] * NUM_CLIENTS # Preallocate list for client PIDs
    pad = None
    sct = None
    gamepads = [None] * NUM_CLIENTS # Preallocate list for gamepads

    try:
        # 1. Create init.txt (Manual step or add code here to create it)
        if not os.path.exists(INIT_TXT_PATH):
             print(f"Error: {INIT_TXT_PATH} not found. Please create it with server settings.")
             # Example content:
             # sv_name AIPlaysHaloDedicated
             # sv_maxplayers 4
             # sv_public 0
             # sv_map beavercreek slayer
             exit() # Or create the file programmatically

        # 2. Start Dedicated Server
        server_proc = start_dedicated_server(HALO_CE_DIR, INIT_TXT_PATH)
        if not server_proc:
            raise RuntimeError("Failed to start dedicated server.")
        print("Waiting for server to initialize...")
        time.sleep(5) # Give server time to start

        # Send HaloConsole window to back
        send_dedicated_server_to_back()

        # 3. Start Halo Client
        def launch_client(i):
            print(f"Starting Halo client {i+1}/{NUM_CLIENTS}...")
            client_proc =  start_halo_client(
                HALO_CE_DIR, MULTICLIENT_EXE, HALOCE_EXE,
                WIN_WIDTH, WIN_HEIGHT, TARGET_FPS, SERVER_IP, SERVER_PORT
            )
            return client_proc
        
        tmp_client_procs = []
        for i in range(NUM_CLIENTS):
            client_proc = launch_client(i)
            tmp_client_procs.append(client_proc)
            if not client_proc:
                raise RuntimeError("Failed to start Halo client.")
            print(f"Halo client {i+1}/{NUM_CLIENTS} started (PID: {client_proc.pid}).")
        client_procs = tmp_client_procs # Update client_procs with successfully launched clients
        assert len(client_procs) == NUM_CLIENTS, "Mismatch in number of client spawns launched and found."
        print("Waiting for clients to launch and connect...")
        time.sleep(15) # Give clients time to launch and connect to server

        tmp_client_pids = []
        for proc in subprocess.check_output("tasklist", shell=True).decode().splitlines():
            if "haloce.exe" in proc:
                pid = int(proc.split()[1])
                tmp_client_pids.append(pid)
                print(f"Found Halo client process (PID: {pid}).")
        client_pids = tmp_client_pids # Update client_pids with found PIDs
        assert len(client_pids) == NUM_CLIENTS, "Mismatch in number of clients launched and found."
        
        # 4. Setup Capture
        # Tile Halo windows using pywin32
        try:
            hwnds = find_halo_windows(NUM_CLIENTS)
            # Get monitor resolution
            screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            tile_windows(hwnds, WIN_WIDTH, WIN_HEIGHT, screen_width, screen_height)
            print(f"Tiled {len(hwnds)} Halo window(s) on screen.")

            # Setup capture for all client windows
            sct, client_monitors = setup_client_captures(hwnds)
            print(f"Setup capture for {len(client_monitors)} client windows.")

            # --- Take one-time screenshot from each capture ---
            images_dir = Path("images")
            images_dir.mkdir(exist_ok=True)
            for idx, monitor in enumerate(client_monitors):
                frame = capture_frame(sct, monitor)
                img_path = images_dir / f"client_{idx+1}.png"
                from PIL import Image
                Image.fromarray(frame).save(img_path)
                print(f"Saved screenshot for client {idx+1} to {img_path}")
        except Exception as e:
            print(f"Error tiling Halo windows or setting up capture: {e}")

        # 5. Load Model (Placeholder)
        model = load_model()
        # if not model:
        #     raise RuntimeError("Failed to load model.") # Enable when implemented

        # 6. Setup Gamepads
        for i in range(NUM_CLIENTS):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            pad = setup_gamepad(init_left_stick=(x, y))
            if pad:
                print(f"Initialized gamepad {i+1} with random left stick: x={x:.2f}, y={y:.2f}")
            else:
                print(f"Error initializing gamepad {i+1}")
            gamepads[i] = pad # Store gamepad in list

        # 7. Main Loop
        print("Starting main loop (capture -> inference -> control)... Press Ctrl+C to stop.")
        frame_time = 1.0 / CAPTURE_FPS
        while True:
            loop_start_time = time.time()

            # Capture for all clients
            frames = [capture_frame(sct, monitor) for monitor in client_monitors]

            # Inference and Control per client
            for idx, (frame, pad) in enumerate(zip(frames, gamepads, strict=True)):
                detections = run_inference(model, frame)
                apply_policy_and_control(detections, pad)

            # Regulate loop speed
            elapsed_time = time.time() - loop_start_time
            sleep_time = frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: Loop took longer ({elapsed_time:.3f}s) than target frame time ({frame_time:.3f}s)")


    except KeyboardInterrupt:
        print("Ctrl+C detected. Shutting down...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        # Terminate processes
        for proc in client_procs:
            if proc is None:
                print("No client process found.")
                continue
            print(f"Terminating client spawn process (PID: {proc.pid})...")
            proc.terminate()
            proc.wait() # Wait for termination
        for pid in client_pids:
            if pid is None:
                print("No client PID found.")
                continue
            try:
                print(f"Terminating client process (PID: {pid})...")
                os.kill(pid, signal.SIGTERM) # Force kill
            except OSError:
                print(f"Client process (PID: {pid}) already terminated.")
        if server_proc:
            print(f"Terminating server process (PID: {server_proc.pid})...")
            server_proc.terminate()
            server_proc.wait() # Wait for termination
        print("Cleanup complete.")
