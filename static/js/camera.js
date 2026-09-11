/**
 * Camera Module for Chapati Analyzer™
 * Provides mobile-first camera capture with environment facingMode,
 * live viewfinder reticle, high-resolution snapshot, flip camera, and retake controls.
 */

export class CameraController {
  constructor(videoElement, canvasElement, reticleElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.reticle = reticleElement;
    this.stream = null;
    this.currentFacingMode = "environment"; // Default to back camera on phones
    this.capturedBlob = null;
    this.capturedDataUrl = null;
  }

  async startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera API (getUserMedia) not supported in this browser environment.");
    }

    this.stopCamera();

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: this.currentFacingMode },
        width: { ideal: 1920, min: 640 },
        height: { ideal: 1080, min: 480 },
      },
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;
      await this.video.play();
      if (this.reticle) this.reticle.style.display = "block";
      return true;
    } catch (err) {
      console.warn("Could not start camera with ideal constraints, trying fallback:", err);
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        this.video.srcObject = this.stream;
        await this.video.play();
        if (this.reticle) this.reticle.style.display = "block";
        return true;
      } catch (fallbackErr) {
        throw new Error(
          fallbackErr.name === "NotAllowedError"
            ? "Camera permission was denied. Please allow camera access or use the file upload option."
            : `Unable to access camera: ${fallbackErr.message}`
        );
      }
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }
    if (this.reticle) {
      this.reticle.style.display = "none";
    }
  }

  async flipCamera() {
    this.currentFacingMode = this.currentFacingMode === "environment" ? "user" : "environment";
    return this.startCamera();
  }

  captureFrame() {
    if (!this.video || !this.video.videoWidth) {
      throw new Error("Camera video stream is not ready for capture.");
    }

    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;

    this.canvas.width = vw;
    this.canvas.height = vh;
    const ctx = this.canvas.getContext("2d");

    // If front camera, mirror image back to natural orientation
    if (this.currentFacingMode === "user") {
      ctx.translate(vw, 0);
      ctx.scale(-1, 1);
    }

    ctx.drawImage(this.video, 0, 0, vw, vh);

    this.capturedDataUrl = this.canvas.toDataURL("image/jpeg", 0.92);
    return this.capturedDataUrl;
  }

  async getCapturedBlob() {
    return new Promise((resolve) => {
      this.canvas.toBlob((blob) => {
        this.capturedBlob = blob;
        resolve(blob);
      }, "image/jpeg", 0.92);
    });
  }
}
