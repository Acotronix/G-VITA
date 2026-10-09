
import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Camera,
  CameraOff,
  Play,
  Square,
  RotateCcw,
  X,
  Zap,
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock,
  ChevronLeft,
} from "lucide-react";
import "../styles/posesession.css";

const POSE_DATA = {
  "01": {
    name: "Anantasana",
    level: "Intermediate",
    duration: 8,
    focus: "Balance · hip mobility",
    description: "A side-reclining balance that invites length through the whole body.",
    cue: "Let your breath set the pace.",
    tips: [
      "Root your bottom hip firmly into the mat.",
      "Extend through your top heel as you lift.",
      "Keep your gaze soft and forward.",
    ],
  },
  "02": {
    name: "Bhujangasana",
    level: "Beginner",
    duration: 6,
    focus: "Chest · gentle extension",
    description: "A gentle, supported backbend that opens the front of the body.",
    cue: "Open your chest with each inhale.",
    tips: [
      "Press the tops of your feet into the mat.",
      "Keep elbows slightly bent, shoulders away from ears.",
      "Gaze forward or slightly upward.",
    ],
  },
  "03": {
    name: "Marjariasana",
    level: "Beginner",
    duration: 7,
    focus: "Spine · mobility",
    description: "A grounding hands-and-knees shape for slow spinal movement.",
    cue: "Move with your breath, not against it.",
    tips: [
      "Stack wrists under shoulders, knees under hips.",
      "Round your spine fully on the exhale.",
      "Let the movement be slow and intentional.",
    ],
  },
  "04": {
    name: "Trikonasana",
    level: "Intermediate",
    duration: 8,
    focus: "Balance · side body",
    description: "A wide, steady triangle with space through both sides of the torso.",
    cue: "Find length before you find depth.",
    tips: [
      "Ground through the outer edge of your back foot.",
      "Reach your top arm directly above your shoulder.",
      "Keep both sides of the waist long.",
    ],
  },
  "05": {
    name: "Vajrasana",
    level: "Beginner",
    duration: 5,
    focus: "Balance · breath",
    description: "A quiet kneeling seat for steady posture and a calmer breath.",
    cue: "Settle into stillness.",
    tips: [
      "Sit on your heels with spine tall.",
      "Rest hands on thighs, palms down.",
      "Soften your jaw and shoulders.",
    ],
  },
  "06": {
    name: "Virabhadrasana I",
    level: "Intermediate",
    duration: 8,
    focus: "Strength · grounding",
    description: "A grounded standing lunge with an expansive, upright reach.",
    cue: "Root down to rise up.",
    tips: [
      "Square your hips toward the front of the mat.",
      "Bend your front knee to 90° over the ankle.",
      "Reach both arms strongly overhead.",
    ],
  },
};

const DEMO_SCORES = [72, 78, 81, 75, 84, 88, 82, 79, 86, 90, 85, 83];

function fmtTime(seconds) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remaining = (seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remaining}`;
}

export default function PoseSession() {
  const { poseId } = useParams();
  const navigate = useNavigate();
  const pose = POSE_DATA[poseId] ?? POSE_DATA["01"];

  const [cameraState, setCameraState] = useState("off");
  const [demoMode, setDemoMode] = useState(false);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [demoScore, setDemoScore] = useState(null);
  const [demoIdx, setDemoIdx] = useState(0);
  const [detectedPose, setDetectedPose] = useState(null);

  // ML results
  const [mlResult, setMlResult] = useState(null);
  const [mlLoading, setMlLoading] = useState(false);
  const [mlError, setMlError] = useState("");

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const demoRef = useRef(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraState("off");
  }, []);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      clearInterval(timerRef.current);
      clearInterval(demoRef.current);
    };
  }, []);

  // Session timer
  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(() => {
        setElapsed((value) => value + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }

    return () => clearInterval(timerRef.current);
  }, [running]);

  // Demo score simulation
  useEffect(() => {
    if (demoMode && running) {
      demoRef.current = setInterval(() => {
        setDemoIdx((index) => (index + 1) % DEMO_SCORES.length);
      }, 1800);
    } else {
      clearInterval(demoRef.current);
    }

    return () => clearInterval(demoRef.current);
  }, [demoMode, running]);

  useEffect(() => {
    if (demoMode) {
      setDemoScore(DEMO_SCORES[demoIdx]);
      setDetectedPose(pose.name);
    }
  }, [demoMode, demoIdx, pose.name]);

  const startCamera = useCallback(async () => {
    setCameraState("requesting");
    setMlError("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraState("active");
    } catch (error) {
      console.error("Camera error:", error);
      setCameraState("denied");
    }
  }, []);

  // Capture a camera frame and send it to the ML backend
  const handleAnalyzePose = useCallback(async () => {
    const video = videoRef.current;

    if (
      !video ||
      video.readyState < 2 ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      setMlError("Start the camera and wait for the video to load.");
      return;
    }

    setMlLoading(true);
    setMlError("");

    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");

      if (!context) {
        throw new Error("Could not capture the camera frame.");
      }

      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve, reject) => {
        canvas.toBlob(
          (image) => {
            if (image) {
              resolve(image);
            } else {
              reject(new Error("Could not create the image."));
            }
          },
          "image/jpeg",
          0.85
        );
      });

      if (blob.size > 8 * 1024 * 1024) {
        throw new Error("Captured image exceeds the 8 MB limit.");
      }

      const response = await fetch(
        "http://localhost:5000/api/ml/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "image/jpeg",
          },
          body: blob,
        }
      );

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The backend returned an invalid response. Check that Node is running."
        );
      }

      if (!response.ok) {
        throw new Error(
          data.detail ||
            data.error ||
            `Pose analysis failed (${response.status}).`
        );
      }

      if (!data.success) {
        throw new Error(
          data.message || "No pose could be detected in this frame."
        );
      }

      setMlResult(data);
      setDetectedPose(data.predicted_pose);
    } catch (error) {
      console.error("ML pose analysis error:", error);
      setMlError(
        error.message || "Could not connect to the ML service."
      );
    } finally {
      setMlLoading(false);
    }
  }, []);

  const handleStartLive = useCallback(async () => {
    setDemoMode(false);
    setDemoScore(null);
    setDetectedPose(null);
    setMlResult(null);
    setMlError("");
    setRunning(false);

    await startCamera();

    // Start timer only if camera startup succeeded.
    // cameraState is updated asynchronously, so check the stream.
    if (streamRef.current) {
      setRunning(true);
    }
  }, [startCamera]);

  const handleTryDemo = useCallback(() => {
    stopCamera();
    setMlResult(null);
    setMlError("");
    setDemoMode(true);
    setRunning(true);
    setDemoIdx(0);
    setDemoScore(DEMO_SCORES[0]);
    setDetectedPose(pose.name);
  }, [stopCamera, pose.name]);

  const handleStop = useCallback(() => {
    setRunning(false);
    setDemoMode(false);
    stopCamera();
    setElapsed(0);
    setDemoScore(null);
    setDetectedPose(null);
    setDemoIdx(0);
    setMlResult(null);
    setMlError("");
    setMlLoading(false);
  }, [stopCamera]);

  const handleCancel = useCallback(() => {
    handleStop();
    navigate("/sessions");
  }, [handleStop, navigate]);

  const isLive = cameraState === "active";
  const isRunning = running;

  const scoreVal = demoMode
    ? demoScore
    : mlResult?.form_analysis?.success
      ? mlResult.form_analysis.result.form_score
      : null;

  const scoreLbl =
    scoreVal === null
      ? "—"
      : scoreVal >= 85
        ? "Good"
        : scoreVal >= 70
          ? "Fair"
          : "Needs work";

  const scoreColor =
    scoreVal === null
      ? "#9b93a5"
      : scoreVal >= 85
        ? "#5f9e7a"
        : scoreVal >= 70
          ? "#b59640"
          : "#b05858";

  return (
    <div className="ps-page">
      {/* Page header */}
      <div className="ps-header">
        <div className="ps-header-left">
          <button
            className="ps-back-btn"
            onClick={handleCancel}
            aria-label="Back to Yoga Sessions"
          >
            <ChevronLeft size={16} />
            Yoga sessions
          </button>

          <div className="ps-eyebrow">A LITTLE SPACE, JUST FOR YOU</div>
          <h1 className="ps-title">Live session</h1>
          <p className="ps-subtitle">
            Today's focus: <strong>{pose.name}</strong>. {pose.cue}
          </p>
        </div>

        <button
          className="ps-cancel-btn"
          onClick={handleCancel}
          aria-label="Cancel session and return"
        >
          <X size={15} />
          Cancel all
        </button>
      </div>

      {/* Main layout */}
      <div className="ps-body">
        {/* Camera panel */}
        <div className="ps-camera-panel">
          <div className="ps-viewport" aria-label="Camera viewport">
            <div className="ps-viewport-bar">
              <div className="ps-cam-status">
                {isLive ? (
                  <>
                    <span className="ps-cam-dot ps-cam-dot--live" />
                    <span>Camera live</span>
                  </>
                ) : demoMode ? (
                  <>
                    <span className="ps-cam-dot ps-cam-dot--demo" />
                    <span>Demo mode</span>
                  </>
                ) : (
                  <>
                    <CameraOff size={13} />
                    <span>Camera off</span>
                  </>
                )}
              </div>

              <span className="ps-frames-note">
                {isLive ? "Frames sent only on request" : "Camera frames are not stored"}
              </span>
            </div>

            <video
              ref={videoRef}
              className={`ps-video${isLive ? " ps-video--visible" : ""}`}
              muted
              playsInline
              aria-label="Live camera feed"
            />

            {!isLive && (
              <div className={`ps-idle${demoMode ? " ps-idle--demo" : ""}`}>
                {demoMode ? (
                  <>
                    <div className="ps-demo-badge">
                      <Zap size={22} />
                    </div>
                    <p className="ps-idle-title">Demo mode active</p>
                    <p className="ps-idle-sub">
                      Simulated pose detection running · no camera required
                    </p>
                  </>
                ) : cameraState === "denied" ? (
                  <>
                    <AlertCircle
                      size={34}
                      className="ps-idle-icon ps-idle-icon--warn"
                    />
                    <p className="ps-idle-title">Camera access denied</p>
                    <p className="ps-idle-sub">
                      Allow camera access in your browser settings and try again.
                    </p>
                  </>
                ) : cameraState === "requesting" ? (
                  <>
                    <div className="ps-spinner" aria-label="Requesting camera" />
                    <p className="ps-idle-title">Requesting camera…</p>
                  </>
                ) : (
                  <>
                    <Camera size={34} className="ps-idle-icon" />
                    <p className="ps-idle-title">
                      Your space, on your terms.
                    </p>
                    <p className="ps-idle-sub">
                      Camera access begins when you choose to start.
                    </p>
                    <p className="ps-idle-privacy">
                      <span className="ps-privacy-dot" />
                      Camera frames are only sent for analysis when requested
                    </p>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="ps-controls">
            <div
              className="ps-timer"
              aria-live="polite"
              aria-label="Session timer"
            >
              <Clock size={14} aria-hidden="true" />
              <span className="ps-timer-digits">{fmtTime(elapsed)}</span>
              <span className="ps-timer-label">elapsed</span>
            </div>

            <div className="ps-ctrl-btns">
              {isRunning ? (
                <button
                  className="ps-btn ps-btn--stop"
                  onClick={handleStop}
                  aria-label="Stop session"
                >
                  <Square size={14} fill="currentColor" />
                  Stop session
                </button>
              ) : (
                <>
                  <button
                    className="ps-btn ps-btn--live"
                    onClick={handleStartLive}
                    aria-label="Start live session with camera"
                    disabled={cameraState === "requesting"}
                  >
                    <Play size={14} fill="currentColor" />
                    Start live session
                  </button>

                  <button
                    className="ps-btn ps-btn--demo"
                    onClick={handleTryDemo}
                    aria-label="Try demo mode without camera"
                  >
                    <Zap size={14} />
                    Try Demo Mode
                  </button>
                </>
              )}

              {isLive && !demoMode && (
                <button
                  className="ps-btn"
                  onClick={handleAnalyzePose}
                  disabled={mlLoading}
                >
                  <Activity size={14} />
                  {mlLoading ? "Analyzing..." : "Analyze Pose"}
                </button>
              )}

              {isRunning && (
                <button
                  className="ps-btn ps-btn--reset"
                  onClick={handleStop}
                  aria-label="Reset session"
                >
                  <RotateCcw size={14} />
                </button>
              )}
            </div>
          </div>

          {/* Camera status message */}
          <p className="ps-cam-note">
            {isLive
              ? "Camera is live. A single frame is sent to your local ML service only when you click Analyze Pose."
              : cameraState === "denied"
                ? "Camera access was denied. Enable it in browser settings."
                : "Camera is off. Start when your space feels ready."}
          </p>
        </div>

        {/* Information panel */}
        <div className="ps-info-panel">
          <div className="ps-pose-header">
            <span className="ps-panel-label">CURRENT POSE</span>

            {isRunning && (
              <span className="ps-running-badge">
                <span className="ps-run-dot" />
                {demoMode ? "Demo" : "Live"}
              </span>
            )}
          </div>

          <h2 className="ps-pose-name">{pose.name}</h2>
          <p className="ps-pose-desc">{pose.description}</p>

          <div className="ps-pose-focus">
            <span className="ps-focus-dot" />
            {pose.focus}
          </div>

          {/* Alignment card */}
          <div className="ps-alignment-card">
            <div className="ps-alignment-label">ALIGNMENT SIGNAL</div>

            {scoreVal !== null ? (
              <div className="ps-score-row">
                <div
                  className="ps-score-bar-wrap"
                  aria-label={`Alignment score ${scoreVal}%`}
                >
                  <div
                    className="ps-score-bar-fill"
                    style={{
                      width: `${scoreVal}%`,
                      background: scoreColor,
                    }}
                  />
                </div>

                <span className="ps-score-num" style={{ color: scoreColor }}>
                  {scoreVal}%
                </span>

                <span className="ps-score-lbl" style={{ color: scoreColor }}>
                  {scoreLbl}
                </span>
              </div>
            ) : (
              <p className="ps-alignment-placeholder">
                Form feedback appears after you analyze a camera frame.
              </p>
            )}

            <p className="ps-alignment-note">
              Geometric feedback only · not a diagnosis
            </p>

            {mlError && (
              <p className="ps-cam-note" role="alert">
                {mlError}
              </p>
            )}

            {mlResult && !demoMode && (
              <div className="ps-ml-results" aria-live="polite">
                <h3>ML Pose Analysis</h3>

                <p>
                  <strong>Detected pose:</strong>{" "}
                  {mlResult.predicted_pose}
                </p>

                <p>
                  <strong>Classification confidence:</strong>{" "}
                  {mlResult.pose_confidence}%
                </p>

                {mlResult.form_analysis?.success ? (
                  <>
                    <p>
                      <strong>Form feedback:</strong>{" "}
                      {mlResult.form_analysis.result.feedback}
                    </p>
                    <p>
                      <strong>Form score:</strong>{" "}
                      {mlResult.form_analysis.result.form_score}/100
                    </p>
                  </>
                ) : (
                  <p>
                    Pose detected, but form analysis failed:{" "}
                    {mlResult.form_analysis?.error ||
                      mlResult.form_analysis?.message ||
                      "Unknown error"}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Detection card */}
          <div className="ps-detection-card">
            <div className="ps-panel-label">DETECTION</div>

            <div className="ps-detection-row">
              <div
                className={`ps-detect-icon${
                  detectedPose ? " ps-detect-icon--active" : ""
                }`}
              >
                <Activity size={16} />
              </div>

              <div>
                <p className="ps-detect-title">
                  {detectedPose
                    ? `Detected: ${detectedPose}`
                    : "Person detection"}
                </p>
                <p className="ps-detect-sub">
                  {detectedPose
                    ? "Pose recognised · alignment active"
                    : "Starts when camera analysis begins"}
                </p>
              </div>
            </div>

            <div className="ps-detect-status">
              <span className="ps-detect-lbl">Frame quality</span>
              <span className="ps-detect-val">
                {isLive
                  ? "Live feed"
                  : demoMode
                    ? "Simulated"
                    : "Waiting for camera"}
              </span>
            </div>

            {!isRunning && (
              <div className="ps-no-label-note">
                <AlertCircle size={13} />
                <span>
                  Start a session, then click Analyze Pose to run the ML pipeline.
                </span>
              </div>
            )}
          </div>

          {/* Practice tips */}
          <div className="ps-tips-card">
            <div className="ps-panel-label">PRACTICE TIPS</div>
            <ul className="ps-tips-list">
              {pose.tips.map((tip, index) => (
                <li key={index} className="ps-tip-item">
                  <CheckCircle2 size={13} className="ps-tip-icon" />
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Privacy note */}
      <div className="ps-privacy-bar">
        <AlertCircle size={12} />
        <span>
          Your camera stays off until you start. Only a frame you request is
          sent to your local ML service for analysis.
        </span>

        {isLive && (
          <button className="ps-turn-off-cam" onClick={stopCamera}>
            Turn off camera
          </button>
        )}
      </div>
    </div>
  );
}
