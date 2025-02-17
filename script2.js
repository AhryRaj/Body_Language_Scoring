import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver } = vision;

// DOM Elements
const demosSection = document.getElementById("demos");
const eyeContactScoreElement = document.getElementById("eye-contact-score");
const headPostureScoreElement = document.getElementById("head-posture-score");
const overallScoreElement = document.getElementById("overall-score");
const enableWebcamButton = document.getElementById("webcamButton");
const videoContainer = document.getElementById("videoContainer");

// Global Variables
let faceLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastVideoTime = -1;

// Scoring Counters
let eyeContactFrames = 0;
let headPostureFrames = 0;
let totalFrames = 0;

// Face Mesh Constants
const LEFT_EYE_INDICES = [33, 133, 159, 145];
const RIGHT_EYE_INDICES = [362, 263, 386, 374];
const LEFT_IRIS_INDICES = [468, 469, 470, 471];
const RIGHT_IRIS_INDICES = [473, 474, 475, 476];
const ADAPTIVE_FRAMES = 30;
const MAX_HEAD_ANGLE = 25; // Degrees

// Video Elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
let eyeContact = false;
let headPosture = false;

// Initialize FaceLandmarker
async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU"
    },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    runningMode: "VIDEO",
    numFaces: 1
  });
}
createFaceLandmarker();

// Webcam Functions
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function enableCam() {
  if (!faceLandmarker) return;

  if (webcamRunning) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE WEBCAM";
    videoContainer.style.display = "none";
    video.srcObject.getTracks().forEach(track => track.stop());
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    calculateFinalScores();
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE WEBCAM";
    videoContainer.style.display = "block";
    
    // Reset counters
    eyeContactFrames = 0;
    headPostureFrames = 0;
    totalFrames = 0;

    const constraints = { 
      video: { 
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      } 
    };
    
    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
      });
  }
}

if (hasGetUserMedia()) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Video Processing
function adjustVideoSize() {
  const container = document.querySelector(".video-container");
  const videoAspectRatio = video.videoWidth / video.videoHeight;
  
  container.style.paddingTop = `${(1 / videoAspectRatio) * 100}%`;
  
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  
  video.style.width = '100%';
  video.style.height = '100%';
}

async function predictWebcam() {
  if (!webcamRunning) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    return;
  }

  adjustVideoSize();
  const startTimeMs = Date.now();
  totalFrames++;
  
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = faceLandmarker.detectForVideo(video, startTimeMs);

    canvasCtx.save();
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (results.faceLandmarks?.length > 0) {
      const landmarks = results.faceLandmarks[0];

      // Start scoring after calibration period
      if (totalFrames > ADAPTIVE_FRAMES) {
        // Enhanced eye contact detection with gaze analysis
        eyeContact = isEyeContact(landmarks) && isDirectGaze(landmarks);
        if (eyeContact) {
          eyeContactFrames++;
        }

        // Head Posture Detection
        headPosture = isHeadPostureStable(results);
        if (headPosture) {
          headPostureFrames++;
        }
      }

    } else {
      // No face detected display
      canvasCtx.restore(); 
      canvasCtx.fillStyle = 'red';
      canvasCtx.font = '60px Arial';
      canvasCtx.fillText('No Face Detected', 10, 60);
    }
    canvasCtx.restore();
  }

  // Update scores and check interval
  updateScoreDisplay();
  window.requestAnimationFrame(predictWebcam);
}

// Detection Functions
function isEyeContact(landmarks) {
  const leftEAR = calculateEAR(landmarks, LEFT_EYE_INDICES);
  const rightEAR = calculateEAR(landmarks, RIGHT_EYE_INDICES);
  const avgEAR = (leftEAR + rightEAR) / 2;
  return avgEAR > 0.23;
}

function calculateEAR(landmarks, eyeIndices) {
  const [p1, p2, p3, p4] = eyeIndices.map(i => landmarks[i]);
  const horizontal = Math.hypot(p1.x - p2.x, p1.y - p2.y);
  const vertical = Math.hypot(p3.x - p4.x, p3.y - p4.y);
  return vertical / (2 * horizontal + 1e-6);
}

function isDirectGaze(landmarks) {
  const gaze = detectGazeDirection(landmarks);
  return gaze.left.horizontal === 'center' && 
         gaze.right.horizontal === 'center' &&
         gaze.left.vertical === 'center' &&
         gaze.right.vertical === 'center';
}

function detectGazeDirection(landmarks) {
  const leftEyeBounds = getEyeBounds(landmarks, LEFT_EYE_INDICES);
  const rightEyeBounds = getEyeBounds(landmarks, RIGHT_EYE_INDICES);
  const leftIris = landmarks[LEFT_IRIS_INDICES[0]];
  const rightIris = landmarks[RIGHT_IRIS_INDICES[0]];

  return {
    left: calculateEyeZone(leftIris, leftEyeBounds),
    right: calculateEyeZone(rightIris, rightEyeBounds)
  };
}

function getEyeBounds(landmarks, indices) {
  const xs = indices.map(i => landmarks[i].x);
  const ys = indices.map(i => landmarks[i].y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys)
  };
}

function calculateEyeZone(iris, bounds) {
  const xThird = (bounds.maxX - bounds.minX) / 3;
  const yThird = (bounds.maxY - bounds.minY) / 3;
  
  const horizontalPos = iris.x - bounds.minX;
  const verticalPos = iris.y - bounds.minY;

  let hDir = 'center';
  let vDir = 'center';

  if (horizontalPos < xThird) hDir = 'left';
  else if (horizontalPos > 2 * xThird) hDir = 'right';
  
  if (verticalPos < yThird) vDir = 'up';
  else if (verticalPos > 2 * yThird) vDir = 'down';

  return { horizontal: hDir, vertical: vDir };
}

function isHeadPostureStable(results) {
  if (!results.facialTransformationMatrixes?.[0]) return false;
  
  const matrix = results.facialTransformationMatrixes[0].data;
  const rotation = matrixToEulerAngles(matrix);
  
  return Math.abs(rotation.pitch) < MAX_HEAD_ANGLE && 
         Math.abs(rotation.yaw) < MAX_HEAD_ANGLE &&
         Math.abs(rotation.roll) < MAX_HEAD_ANGLE;
}

function matrixToEulerAngles(matrix) {
  const pitch = Math.atan2(matrix[9], matrix[10]) * (180/Math.PI);
  const yaw = Math.atan2(-matrix[8], Math.sqrt(matrix[9]**2 + matrix[10]**2)) * (180/Math.PI);
  const roll = Math.atan2(matrix[4], matrix[0]) * (180/Math.PI);
  return { pitch, yaw, roll };
}

// Score Display
function updateScoreDisplay() {
  const effectiveFrames = Math.max(totalFrames - ADAPTIVE_FRAMES, 1);
  const eyePercent = totalFrames > ADAPTIVE_FRAMES 
    ? (eyeContactFrames / effectiveFrames * 100)
    : 0;
  
  const headPercent = totalFrames > ADAPTIVE_FRAMES
    ? (headPostureFrames / effectiveFrames * 100)
    : 0;

  const overall = (eyePercent + headPercent) / 2;

  eyeContactScoreElement.textContent = `${Math.min(eyePercent, 100).toFixed(1)}%`;
  headPostureScoreElement.textContent = `${Math.min(headPercent, 100).toFixed(1)}%`;
  overallScoreElement.textContent = `${Math.min(overall, 100).toFixed(1)}%`;
}

function calculateFinalScores() {
  updateScoreDisplay();
}