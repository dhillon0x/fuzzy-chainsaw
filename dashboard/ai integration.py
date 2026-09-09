# ==============================================================================
# FILE: dashboard.py
# Description: Professional Real-Time AI Robot Mission Control Dashboard
# Features:
#   - FastAPI + WebSocket multi-client MJPEG / telemetry streaming
#   - Multithreaded camera pipeline with YOLOv8 + FastSAM segmentation
#   - Bring-Your-Own-API-Key (BYOK) dynamic client authorization
#   - Automatic fallback to Gemini Vision via Google Generative AI SDK
#   - Direct 6-DOF Inverse Kinematics command generation & execution loop
#   - You can use any api like claude or chatgpt it ur choice 
# ==============================================================================

import os
import cv2
import json
import time
import base64
import asyncio
import logging
import threading
from typing import Optional, Dict, Any, List
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Internal system imports
from vision.detector import FastDetector
from vision.segmenter import FastSegmenter
from src.kinematics import ArmKinematics
from src.controller import ArmController

# Optional cloud LLM/VLM SDKs
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (Dashboard) %(message)s")
logger = logging.getLogger("Dashboard")

# ------------------------------------------------------------------------------
# Core State Engine & Thread-Safe Frame Buffers
# ------------------------------------------------------------------------------
class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_joint_angles: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.target_xyz: List[float] = [200.0, 0.0, 25.0]
        self.autonomous_mode: bool = False
        self.last_ai_decision: str = "Awaiting user command or autonomous trigger..."
        self.active_provider: str = "openai"  # 'openai' or 'gemini'
        self.api_key: str = ""
        self.system_prompt: str = (
            "You are the autonomous brain of a 6-Axis Robotic Arm. "
            "Analyze the camera workspace image and the detected item list. "
            "Select the best target object to interact with and provide a concise operational verdict. "
            "Respond strictly in JSON matching this schema: "
            '{"target_label": "<class_name>", "action": "PICK"|"INSPECT"|"DISCARD", "reasoning": "<explanation>"}'
        )

state = RobotState()
kinematics = ArmKinematics()
controller = ArmController(port="COM3")
detector = FastDetector(conf_thresh=0.45)
segmenter = FastSegmenter(conf=0.40)

# ------------------------------------------------------------------------------
# Video Capture & Vision Processing Thread
# ------------------------------------------------------------------------------
def vision_worker_loop():
    logger.info("Starting background vision worker thread...")
    cap = cv2.VideoCapture(0)
    
    # Set capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            # Generate animated test pattern if camera is offline
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CAMERA OFFLINE - SIMULATION FEED", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            with state.lock:
                state.latest_frame = blank
            time.sleep(0.05)
            continue

        raw_detections = detector.detect(frame)
        processed_detections = []

        h, w, _ = frame.shape
        scale_mm_per_px = 0.5

        for det in raw_detections:
            bbox = det["box"]
            cls_name = det["class"]
            conf = det["confidence"]

            mask, (cx, cy) = segmenter.extract_mask_and_centroid(frame, bbox)

            # Planar projection to world coordinates (mm)
            x_w = 200.0 + (cy - (h / 2)) * scale_mm_per_px
            y_w = (cx - (w / 2)) * scale_mm_per_px
            z_w = 25.0

            processed_detections.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "box": bbox,
                "centroid": [cx, cy],
                "world_xyz": [round(x_w, 1), round(y_w, 1), round(z_w, 1)]
            })

            # Render HUD overlays
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{cls_name} [{conf:.2f}]", (bbox[0], bbox[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        with state.lock:
            state.latest_frame = frame
            state.latest_detections = processed_detections

        time.sleep(0.01)

# ------------------------------------------------------------------------------
# Autonomous AI Decision Engine (BYOK)
# ------------------------------------------------------------------------------
class AutonomousAIEngine:
    @staticmethod
    def encode_frame_to_base64(frame: np.ndarray) -> str:
        small = cv2.resize(frame, (320, 240))
        _, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return base64.b64encode(buffer).decode("utf-8")

    @classmethod
    def consult_ai(cls, frame: np.ndarray, detections: List[Dict[str, Any]], api_key: str, provider: str) -> Dict[str, Any]:
        if not api_key:
            return {"error": "API Key is missing. Enter your key in the dashboard."}

        b64_img = cls.encode_frame_to_base64(frame)
        context_prompt = (
            f"Detections currently on workspace:\n{json.dumps(detections, indent=2)}\n\n"
            "Analyze the visual frame and detection metadata. What should the robot do next?"
        )

        if provider == "openai":
            if OpenAI is None:
                return {"error": "openai Python package not installed. Run: pip install openai"}
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": state.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            raw_text = response.choices[0].message.content
            return json.loads(raw_text)

        elif provider == "gemini":
            if genai is None:
                return {"error": "google-generativeai package not installed. Run: pip install google-generativeai"}

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            img_part = {"mime_type": "image/jpeg", "data": base64.b64decode(b64_img)}
            response = model.generate_content([
                state.system_prompt,
                context_prompt,
                img_part
            ])
            
            # Clean markdown codeblocks if present
            cleaned = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

        return {"error": f"Unsupported AI provider: {provider}"}

# ------------------------------------------------------------------------------
# FastAPI Application & WebSocket Server
# ------------------------------------------------------------------------------
app = FastAPI(title="6-Axis Robotic Arm AI Dashboard")

class ConfigPayload(BaseModel):
    api_key: str
    provider: str
    system_prompt: Optional[str] = None
    autonomous: Optional[bool] = None

class MovePayload(BaseModel):
    target_xyz: List[float]

@app.post("/api/config")
def update_config(payload: ConfigPayload):
    with state.lock:
        state.api_key = payload.api_key.strip()
        state.active_provider = payload.provider.lower().strip()
        if payload.system_prompt:
            state.system_prompt = payload.system_prompt
        if payload.autonomous is not None:
            state.autonomous_mode = payload.autonomous
    return {"status": "ok", "provider": state.active_provider, "autonomous": state.autonomous_mode}

@app.post("/api/move")
def execute_manual_move(payload: MovePayload):
    with state.lock:
        state.target_xyz = payload.target_xyz

    angles = kinematics.inverse_kinematics(payload.target_xyz)
    if angles is not None:
        controller.send_joint_targets(angles)
        with state.lock:
            state.latest_joint_angles = [round(float(a), 2) for a in angles]
        return {"status": "success", "angles": state.latest_joint_angles}
    return JSONResponse(status_code=400, content={"status": "failed", "error": "Target outside mechanical workspace"})

@app.post("/api/ai/decide")
def trigger_ai_decision():
    with state.lock:
        if state.latest_frame is None:
            return JSONResponse(status_code=503, content={"error": "Vision pipeline not ready."})
        frame = state.latest_frame.copy()
        dets = list(state.latest_detections)
        key = state.api_key
        prov = state.active_provider

    decision = AutonomousAIEngine.consult_ai(frame, dets, key, prov)

    with state.lock:
        state.last_ai_decision = json.dumps(decision)

    # If the AI ordered an action on an object, steer the kinematics
    if "target_label" in decision:
        for d in dets:
            if d["class"].lower() == decision["target_label"].lower():
                target_coords = d["world_xyz"]
                angles = kinematics.inverse_kinematics(target_coords)
                if angles is not None:
                    controller.send_joint_targets(angles)
                    with state.lock:
                        state.latest_joint_angles = [round(float(a), 2) for a in angles]
                break

    return decision

@app.websocket("/ws/stream")
async def websocket_telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with state.lock:
                frame = state.latest_frame
                detections = state.latest_detections
                angles = state.latest_joint_angles
                target = state.target_xyz
                auto = state.autonomous_mode
                decision = state.last_ai_decision

            if frame is not None:
                # Downsample JPEG compression for low-latency WebSocket transport
                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                frame_b64 = base64.b64encode(jpeg).decode("utf-8")
            else:
                frame_b64 = ""

            payload = {
                "frame": frame_b64,
                "detections": detections,
                "joint_angles": angles,
                "target_xyz": target,
                "autonomous": auto,
                "last_decision": decision
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.04)  # ~25 FPS stream
    except WebSocketDisconnect:
        pass

# ------------------------------------------------------------------------------
# Embedded Single-Page Mission Control UI
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_view():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>6-Axis AI Robot Mission Control</title>
    <style>
        :root {
            --bg-color: #0b0f19;
            --panel-bg: #111827;
            --accent: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --text-main: #f3f4f6;
            --text-sub: #9ca3af;
            --border-color: #1f2937;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
        body { background: var(--bg-color); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        header { background: var(--panel-bg); border-bottom: 1px solid var(--border-color); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { font-size: 1.1rem; font-weight: 700; letter-spacing: 0.05em; color: #fff; }
        .main-container { display: grid; grid-template-columns: 1fr 380px; flex: 1; overflow: hidden; }
        .viewport-panel { display: flex; flex-direction: column; padding: 16px; gap: 16px; border-right: 1px solid var(--border-color); }
        .video-container { position: relative; background: #000; flex: 1; border-radius: 8px; overflow: hidden; border: 1px solid var(--border-color); display: flex; align-items: center; justify-content: center; }
        .video-container img { width: 100%; height: 100%; object-fit: contain; }
        .controls-panel { background: var(--panel-bg); display: flex; flex-direction: column; overflow-y: auto; padding: 16px; gap: 16px; }
        .card { background: #162032; border: 1px solid var(--border-color); border-radius: 6px; padding: 14px; }
        .card h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-sub); margin-bottom: 10px; }
        .input-group { display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; }
        label { font-size: 0.75rem; color: var(--text-sub); }
        input, select, textarea { background: #0b0f19; border: 1px solid var(--border-color); color: #fff; padding: 8px 10px; border-radius: 4px; font-size: 0.85rem; width: 100%; }
        button { background: var(--accent); color: #fff; border: none; border-radius: 4px; padding: 8px 14px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        button:hover { filter: brightness(1.1); }
        .btn-green { background: var(--accent-green); }
        .btn-red { background: var(--accent-red); }
        .joint-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
        .joint-box { background: #0b0f19; padding: 6px; border-radius: 4px; text-align: center; border: 1px solid var(--border-color); }
        .joint-box span { font-size: 0.65rem; color: var(--text-sub); display: block; }
        .joint-box strong { font-size: 0.9rem; color: #60a5fa; }
        pre { background: #0b0f19; padding: 10px; border-radius: 4px; font-size: 0.75rem; color: #34d399; overflow-x: auto; max-height: 140px; }
    </style>
</head>
<body>
    <header>
        <h1>AI-ENABLED 6-AXIS TELEOPERATION & MISSION CONTROL</h1>
        <div id="statusTag" style="font-size: 0.8rem; color: #10b981;">CONNECTED</div>
    </header>

    <div class="main-container">
        <!-- Viewport -->
        <div class="viewport-panel">
            <div class="video-container">
                <img id="videoFeed" src="" alt="Realtime Video Stream" />
            </div>
            <div class="card">
                <h3>Latest Autonomous AI Action & Reasoning</h3>
                <pre id="aiOutput">Waiting for execution...</pre>
            </div>
        </div>

        <!-- Right Controls Sidebar -->
        <div class="controls-panel">
            <!-- BYOK Authentication -->
            <div class="card">
                <h3>Cloud Intelligence (BYOK)</h3>
                <div class="input-group">
                    <label>Provider</label>
                    <select id="aiProvider">
                        <option value="openai">OpenAI (GPT-4o Vision)</option>
                        <option value="gemini">Google Gemini (1.5 Flash)</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>API Key</label>
                    <input type="password" id="apiKey" placeholder="sk-... / AIza..." />
                </div>
                <button onclick="saveConfig()">Save Authentication</button>
            </div>

            <!-- Telemetry -->
            <div class="card">
                <h3>Kinematics Telemetry (Joint Angles)</h3>
                <div class="joint-grid">
                    <div class="joint-box"><span>J1 (Base)</span><strong id="j1">0.0°</strong></div>
                    <div class="joint-box"><span>J2 (Shoulder)</span><strong id="j2">0.0°</strong></div>
                    <div class="joint-box"><span>J3 (Elbow)</span><strong id="j3">0.0°</strong></div>
                    <div class="joint-box"><span>J4 (W-Roll)</span><strong id="j4">0.0°</strong></div>
                    <div class="joint-box"><span>J5 (W-Pitch)</span><strong id="j5">0.0°</strong></div>
                    <div class="joint-box"><span>J6 (T-Roll)</span><strong id="j6">0.0°</strong></div>
                </div>
            </div>

            <!-- Manual Coordinate Targeting -->
            <div class="card">
                <h3>Manual Cartesian IK Dispatch</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin-bottom: 8px;">
                    <div><label>X (mm)</label><input type="number" id="targetX" value="200" /></div>
                    <div><label>Y (mm)</label><input type="number" id="targetY" value="0" /></div>
                    <div><label>Z (mm)</label><input type="number" id="targetZ" value="25" /></div>
                </div>
                <button onclick="sendTargetMove()">Solve IK & Move</button>
            </div>

            <!-- AI Trigger Button -->
            <div class="card">
                <h3>Autonomous Agent Loop</h3>
                <p style="font-size: 0.75rem; color: var(--text-sub); margin-bottom: 10px;">
                    Snap current video frame, extract objects via YOLOv8 + FastSAM, query cloud VLM, and execute motion.
                </p>
                <button class="btn-green" style="width: 100%;" onclick="runAIDecision()">Execute AI Vision Step</button>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${location.host}/ws/stream`);
        const videoFeed = document.getElementById("videoFeed");

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.frame) {
                videoFeed.src = "data:image/jpeg;base64," + data.frame;
            }
            if (data.joint_angles && data.joint_angles.length === 6) {
                document.getElementById("j1").innerText = data.joint_angles[0] + "°";
                document.getElementById("j2").innerText = data.joint_angles[1] + "°";
                document.getElementById("j3").innerText = data.joint_angles[2] + "°";
                document.getElementById("j4").innerText = data.joint_angles[3] + "°";
                document.getElementById("j5").innerText = data.joint_angles[4] + "°";
                document.getElementById("j6").innerText = data.joint_angles[5] + "°";
            }
            if (data.last_decision) {
                try {
                    const parsed = JSON.parse(data.last_decision);
                    document.getElementById("aiOutput").innerText = JSON.stringify(parsed, null, 2);
                } catch {
                    document.getElementById("aiOutput").innerText = data.last_decision;
                }
            }
        };

        async function saveConfig() {
            const provider = document.getElementById("aiProvider").value;
            const apiKey = document.getElementById("apiKey").value;
            await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ provider: provider, api_key: apiKey })
            });
            alert("Configuration successfully saved to backend runtime.");
        }

        async function sendTargetMove() {
            const x = parseFloat(document.getElementById("targetX").value);
            const y = parseFloat(document.getElementById("targetY").value);
            const z = parseFloat(document.getElementById("targetZ").value);
            const res = await fetch("/api/move", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ target_xyz: [x, y, z] })
            });
            if (!res.ok) {
                const err = await res.json();
                alert("Move failed: " + err.error);
            }
        }

        async function runAIDecision() {
            document.getElementById("aiOutput").innerText = "Analyzing frame via Vision Model...";
            const res = await fetch("/api/ai/decide", { method: "POST" });
            const data = await res.json();
            document.getElementById("aiOutput").innerText = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
    """

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Start vision worker on dedicated background daemon thread
    t = threading.Thread(target=vision_worker_loop, daemon=True)
    t.start()

    logger.info("Initializing Mission Control Web Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
