/**
 * Main Application Logic for Chapati Analyzer™ PWA
 * Orchestrates navigation, camera, file dropzone, REST API communication,
 * visualizations, metric cards, and history.
 */

import { CameraController } from "./camera.js";

class ChapatiApp {
  constructor() {
    this.currentAnalysisData = null;
    this.pendingImagePayload = null; // { type: 'blob'|'base64', data: ... }
    this.cameraController = null;
    this.deferredInstallPrompt = null;

    this.initElements();
    this.initPWA();
    this.initNavigation();
    this.initCameraAndUpload();
    this.initResultsActions();
    this.initHistory();
  }

  initElements() {
    // Nav elements
    this.navButtons = document.querySelectorAll(".nav-btn");
    this.screens = document.querySelectorAll(".screen-view");
    this.brandBtn = document.getElementById("brandBtn");
    this.heroCtaBtn = document.getElementById("heroCtaBtn");

    // PWA Install
    this.installBanner = document.getElementById("installBanner");
    this.installBtn = document.getElementById("installBtn");

    // Camera & Upload elements
    this.tabCameraMode = document.getElementById("tabCameraMode");
    this.tabUploadMode = document.getElementById("tabUploadMode");
    this.cameraViewContainer = document.getElementById("cameraViewContainer");
    this.uploadDropzone = document.getElementById("uploadDropzone");
    this.filePicker = document.getElementById("filePicker");
    this.browseFileBtn = document.getElementById("browseFileBtn");
    this.quickUploadBtn = document.getElementById("quickUploadBtn");
    this.cameraVideo = document.getElementById("cameraVideo");
    this.cameraCanvas = document.getElementById("cameraCanvas");
    this.viewfinderReticle = document.getElementById("viewfinderReticle");
    this.shutterBtn = document.getElementById("shutterBtn");
    this.flipCameraBtn = document.getElementById("flipCameraBtn");

    // Preview & Processing
    this.previewCard = document.getElementById("previewCard");
    this.previewImg = document.getElementById("previewImg");
    this.retakeBtn = document.getElementById("retakeBtn");
    this.confirmAnalyzeBtn = document.getElementById("confirmAnalyzeBtn");
    this.processingOverlay = document.getElementById("processingOverlay");
    this.processingStepMsg = document.getElementById("processingStepMsg");

    // Results elements
    this.resSpecimenId = document.getElementById("resSpecimenId");
    this.resGeometricType = document.getElementById("resGeometricType");
    this.resScoreVal = document.getElementById("resScoreVal");
    this.resVerdictTitle = document.getElementById("resVerdictTitle");
    this.resVerdictQuote = document.getElementById("resVerdictQuote");
    this.visTabButtons = document.querySelectorAll(".vis-tab-btn");
    this.activeVisImage = document.getElementById("activeVisImage");
    this.visCaption = document.getElementById("visCaption");

    // Shape Diagnostics elements
    this.diagTypePill = document.getElementById("diagTypePill");
    this.diagCircularity = document.getElementById("diagCircularity");
    this.diagCircleDev = document.getElementById("diagCircleDev");
    this.diagStability = document.getElementById("diagStability");
    this.diagSymmetry = document.getElementById("diagSymmetry");
    this.diagEllipse = document.getElementById("diagEllipse");
    this.diagSmoothness = document.getElementById("diagSmoothness");
    this.diagCorner = document.getElementById("diagCorner");
    this.diagSolidity = document.getElementById("diagSolidity");
    this.diagCapNotice = document.getElementById("diagCapNotice");
    this.diagCapReason = document.getElementById("diagCapReason");

    // Metric Cards
    this.cardCircularity = document.getElementById("cardCircularity");
    this.barCircularity = document.getElementById("barCircularity");
    this.cardStability = document.getElementById("cardStability");
    this.barStability = document.getElementById("barStability");
    this.cardSymmetry = document.getElementById("cardSymmetry");
    this.barSymmetry = document.getElementById("barSymmetry");
    this.cardSmoothness = document.getElementById("cardSmoothness");
    this.barSmoothness = document.getElementById("barSmoothness");
    this.cardCenterAcc = document.getElementById("cardCenterAcc");
    this.barCenterAcc = document.getElementById("barCenterAcc");
    this.cardBurnControl = document.getElementById("cardBurnControl");
    this.barBurnControl = document.getElementById("barBurnControl");
    this.cardBurnDesc = document.getElementById("cardBurnDesc");
    this.cardTexture = document.getElementById("cardTexture");
    this.barTexture = document.getElementById("barTexture");
    this.cardArea = document.getElementById("cardArea");
    this.cardPerimeter = document.getElementById("cardPerimeter");
    this.cardDiameter = document.getElementById("cardDiameter");
    this.cardAspect = document.getElementById("cardAspect");

    // Action Buttons
    this.btnDownloadReport = document.getElementById("btnDownloadReport");
    this.btnExportJson = document.getElementById("btnExportJson");
    this.btnAnalyzeAnother = document.getElementById("btnAnalyzeAnother");

    // History elements
    this.historyTable = document.getElementById("historyTable");
    this.historyTableBody = document.getElementById("historyTableBody");
    this.emptyHistoryBox = document.getElementById("emptyHistoryBox");
    this.btnExportCsv = document.getElementById("btnExportCsv");

    // Toast Container
    this.toastContainer = document.getElementById("toastContainer");
  }

  initPWA() {
    // Register Service Worker
    if ("serviceWorker" in navigator) {
      window.addEventListener("load", () => {
        navigator.serviceWorker
          .register("/sw.js")
          .then((reg) => console.log("ServiceWorker registered successfully:", reg.scope))
          .catch((err) => console.warn("ServiceWorker registration failed:", err));
      });
    }

    // Install prompt event
    window.addEventListener("beforeinstallprompt", (e) => {
      e.preventDefault();
      this.deferredInstallPrompt = e;
      if (this.installBanner) {
        this.installBanner.style.display = "block";
      }
    });

    if (this.installBtn) {
      this.installBtn.addEventListener("click", async () => {
        if (!this.deferredInstallPrompt) return;
        this.deferredInstallPrompt.prompt();
        const { outcome } = await this.deferredInstallPrompt.userChoice;
        console.log(`PWA install prompt outcome: ${outcome}`);
        this.deferredInstallPrompt = null;
        this.installBanner.style.display = "none";
      });
    }
  }

  initNavigation() {
    this.navButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetScreenId = btn.dataset.screen;
        this.navigateTo(targetScreenId);
      });
    });

    this.brandBtn.addEventListener("click", () => this.navigateTo("screenHome"));
    this.heroCtaBtn.addEventListener("click", () => this.navigateTo("screenAnalyze"));
  }

  navigateTo(screenId) {
    this.screens.forEach((screen) => {
      screen.classList.remove("active-screen");
    });
    this.navButtons.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.screen === screenId);
    });

    const activeScreen = document.getElementById(screenId);
    if (activeScreen) {
      activeScreen.classList.add("active-screen");
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    // Camera management during navigation
    if (screenId === "screenAnalyze") {
      // If on camera tab, start camera
      if (this.tabCameraMode.classList.contains("active-tab") && !this.previewCard.style.display.includes("block")) {
        this.startCameraStream();
      }
    } else {
      if (this.cameraController) {
        this.cameraController.stopCamera();
      }
    }

    // Refresh history if navigating to history
    if (screenId === "screenHistory") {
      this.loadHistory();
    }
  }

  initCameraAndUpload() {
    this.cameraController = new CameraController(
      this.cameraVideo,
      this.cameraCanvas,
      this.viewfinderReticle
    );

    // Tab switcher: Camera vs Upload
    this.tabCameraMode.addEventListener("click", () => {
      this.tabCameraMode.classList.add("active-tab");
      this.tabUploadMode.classList.remove("active-tab");
      this.cameraViewContainer.style.display = "block";
      this.uploadDropzone.style.display = "none";
      this.previewCard.style.display = "none";
      this.startCameraStream();
    });

    this.tabUploadMode.addEventListener("click", () => {
      this.tabUploadMode.classList.add("active-tab");
      this.tabCameraMode.classList.remove("active-tab");
      this.cameraViewContainer.style.display = "none";
      this.uploadDropzone.style.display = "block";
      this.previewCard.style.display = "none";
      this.cameraController.stopCamera();
    });

    this.quickUploadBtn.addEventListener("click", () => {
      this.tabUploadMode.click();
    });

    // Shutter button
    this.shutterBtn.addEventListener("click", async () => {
      try {
        const dataUrl = this.cameraController.captureFrame();
        const blob = await this.cameraController.getCapturedBlob();
        this.cameraController.stopCamera();

        this.pendingImagePayload = { type: "blob", data: blob, dataUrl: dataUrl };
        this.showPreview(dataUrl);
      } catch (err) {
        this.showToast(err.message, "error");
      }
    });

    // Flip camera
    this.flipCameraBtn.addEventListener("click", async () => {
      try {
        await this.cameraController.flipCamera();
      } catch (err) {
        this.showToast(err.message, "error");
      }
    });

    // File Picker & Dropzone
    this.browseFileBtn.addEventListener("click", () => this.filePicker.click());
    this.uploadDropzone.addEventListener("click", (e) => {
      if (e.target !== this.browseFileBtn) this.filePicker.click();
    });

    this.filePicker.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) this.handleSelectedFile(file);
    });

    // Drag & Drop
    ["dragenter", "dragover"].forEach((eventName) => {
      this.uploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.uploadDropzone.classList.add("drag-over");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      this.uploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.uploadDropzone.classList.remove("drag-over");
      });
    });

    this.uploadDropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) this.handleSelectedFile(files[0]);
    });

    // Retake button
    this.retakeBtn.addEventListener("click", () => {
      this.previewCard.style.display = "none";
      this.pendingImagePayload = null;
      if (this.tabCameraMode.classList.contains("active-tab")) {
        this.cameraViewContainer.style.display = "block";
        this.startCameraStream();
      } else {
        this.uploadDropzone.style.display = "block";
      }
    });

    // Confirm & Execute Analysis
    this.confirmAnalyzeBtn.addEventListener("click", () => {
      if (!this.pendingImagePayload) {
        this.showToast("No image selected for inspection.", "error");
        return;
      }
      this.executeAnalysis(this.pendingImagePayload);
    });
  }

  async startCameraStream() {
    try {
      await this.cameraController.startCamera();
    } catch (err) {
      console.warn("Camera auto-start failed:", err);
      // If camera unavailable or denied, automatically switch to upload tab
      this.tabUploadMode.click();
      this.showToast(err.message, "error");
    }
  }

  handleSelectedFile(file) {
    if (!file.type.match(/image\/(jpeg|png|webp)/)) {
      this.showToast("Unsupported file format. Please upload JPEG, PNG, or WebP.", "error");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      this.showToast("File exceeds 20MB limit.", "error");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      this.pendingImagePayload = { type: "blob", data: file, dataUrl: dataUrl };
      this.showPreview(dataUrl);
    };
    reader.readAsDataURL(file);
  }

  showPreview(dataUrl) {
    this.cameraViewContainer.style.display = "none";
    this.uploadDropzone.style.display = "none";
    this.previewImg.src = dataUrl;
    this.previewCard.style.display = "block";
    this.previewCard.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  async executeAnalysis(payload) {
    this.previewCard.style.display = "none";
    this.processingOverlay.style.display = "block";

    // Animated laboratory step messages for technical realism
    const labMessages = [
      "Normalizing color space & smoothing speckle noise...",
      "Extracting dough boundary candidates via Otsu & CIE Lab...",
      "Evaluating circularity, perimeter, and Green's theorem area...",
      "Sampling 360-angle radial rays & testing rotational symmetry...",
      "Calculating Taubin algebraic circle fit & centroid deviation...",
      "Analyzing 5 concentric Maillard toasting zones...",
      "Computing surface Laplacian variance & texture density...",
      "Synthesizing Chapati Perfection Index™..."
    ];
    let msgIdx = 0;
    const msgInterval = setInterval(() => {
      msgIdx = (msgIdx + 1) % labMessages.length;
      this.processingStepMsg.textContent = labMessages[msgIdx];
    }, 450);

    const formData = new FormData();
    if (payload.type === "blob") {
      formData.append("image", payload.data, "chapati_specimen.jpg");
    }

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      clearInterval(msgInterval);
      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || "Analysis failed. Please ensure chapati is clearly visible.");
      }

      this.processingOverlay.style.display = "none";
      this.displayResults(result.data);
      this.navigateTo("screenResults");
      this.showToast("Specimen analysis completed successfully!", "success");

    } catch (err) {
      clearInterval(msgInterval);
      this.processingOverlay.style.display = "none";
      this.previewCard.style.display = "block";
      this.showToast(err.message, "error");
    }
  }

  displayResults(data) {
    this.currentAnalysisData = data;

    const geomType = data.radial.geometric_type || data.score.geometric_type || "ROUND / IRREGULAR";

    // Header
    this.resSpecimenId.textContent = `SPECIMEN ID: ${data.id}  |  ${data.timestamp}`;
    this.resGeometricType.textContent = `GEOMETRIC TYPE: ${geomType}`;
    this.resScoreVal.textContent = data.score.perfection_index.toFixed(1);
    this.resVerdictTitle.textContent = data.score.verdict;
    this.resVerdictQuote.textContent = `"${data.score.verdict_quote}"`;

    // Shape Diagnostics
    this.diagTypePill.textContent = geomType;
    const circPct = (data.geometry.circularity * 100).toFixed(1);
    this.diagCircularity.textContent = `${circPct}%`;

    const devScore = (data.radial.ideal_circle_deviation_score || 100.0).toFixed(1);
    const rmsErrPct = ((data.radial.ideal_circle_rms_error || 0.0) * 100.0).toFixed(1);
    this.diagCircleDev.textContent = `${devScore}% (RMS ${rmsErrPct}%)`;

    const stabPct = data.radial.radius_stability_score.toFixed(1);
    this.diagStability.textContent = `${stabPct}%`;

    const symmPct = data.radial.symmetry_score.toFixed(1);
    this.diagSymmetry.textContent = `${symmPct}%`;

    const ellipseScore = (data.geometry.ellipse_roundness_score || 100.0).toFixed(1);
    const axisRatio = (data.geometry.ellipse_axis_ratio || 1.0).toFixed(2);
    this.diagEllipse.textContent = `${ellipseScore}% (Ratio ${axisRatio})`;

    const smoothPct = (data.radial.boundary_smoothness_score || data.radial.edge_smoothness_score || 100.0).toFixed(1);
    this.diagSmoothness.textContent = `${smoothPct}%`;

    const cornerStr = data.radial.corner_strength || "LOW";
    this.diagCorner.textContent = cornerStr;
    if (cornerStr === "HIGH") {
      this.diagCorner.style.color = "var(--rose-accent)";
    } else if (cornerStr === "MODERATE") {
      this.diagCorner.style.color = "var(--amber-primary)";
    } else {
      this.diagCorner.style.color = "var(--green-accent)";
    }

    const solidityVal = (data.geometry.solidity || 1.0).toFixed(3);
    this.diagSolidity.textContent = solidityVal;

    // Geometric Quality Gate Notice
    if (data.score.score_cap_applied) {
      this.diagCapNotice.style.display = "block";
      this.diagCapReason.textContent = data.score.score_cap_reason || "Non-circular geometry detected. Score capped.";
    } else {
      this.diagCapNotice.style.display = "none";
    }

    // Visualizer tabs
    this.setupVisualizerTabs(data.visualizations);

    // Metric Cards
    this.cardCircularity.textContent = `${circPct}%`;
    this.barCircularity.style.width = `${Math.min(100, circPct)}%`;

    this.cardStability.textContent = `${stabPct}%`;
    this.barStability.style.width = `${Math.min(100, stabPct)}%`;

    this.cardSymmetry.textContent = `${symmPct}%`;
    this.barSymmetry.style.width = `${Math.min(100, symmPct)}%`;

    this.cardSmoothness.textContent = `${smoothPct}%`;
    this.barSmoothness.style.width = `${Math.min(100, smoothPct)}%`;

    const centerPct = data.geometry.center_accuracy_score.toFixed(1);
    this.cardCenterAcc.textContent = `${centerPct}%`;
    this.barCenterAcc.style.width = `${Math.min(100, centerPct)}%`;

    const burnCtrlPct = data.browning.browning_control_score.toFixed(1);
    const burnRatioPct = (data.browning.burn_ratio * 100).toFixed(1);
    this.cardBurnControl.textContent = `${burnCtrlPct}%`;
    this.barBurnControl.style.width = `${Math.min(100, burnCtrlPct)}%`;
    this.cardBurnDesc.textContent = `Burn Ratio: ${burnRatioPct}% | Pattern: ${data.browning.classification}`;

    const textPct = data.texture.texture_score.toFixed(1);
    this.cardTexture.textContent = `${textPct}%`;
    this.barTexture.style.width = `${Math.min(100, textPct)}%`;

    this.cardArea.textContent = Math.round(data.geometry.area).toLocaleString();
    this.cardPerimeter.textContent = Math.round(data.geometry.perimeter).toLocaleString();
    this.cardDiameter.textContent = data.geometry.equivalent_diameter.toFixed(1);
    this.cardAspect.textContent = data.geometry.aspect_ratio.toFixed(2);
  }

  setupVisualizerTabs(visualizations) {
    const captions = {
      ideal_overlay: "Green: Detected contour boundary | Amber: Ideal fitted circle | Red vector: Mass centroid offset",
      browning_heatmap: "Thermal Maillard density heatmap with 5 concentric radial zone demarcations",
      radial_profile: "Continuous 360-angle radial signal r(θ) vs ideal circle radius with deviation shading",
      original: "Unmodified raw camera specimen capture",
      annotated_report: "Complete certified technical laboratory summary card",
    };

    // Default to ideal_overlay
    this.activeVisImage.src = visualizations.ideal_overlay;
    this.visCaption.textContent = captions.ideal_overlay;

    this.visTabButtons.forEach((tab) => {
      tab.classList.remove("active-vis-tab");
      if (tab.dataset.vis === "ideal_overlay") tab.classList.add("active-vis-tab");

      // Replace click handler
      tab.onclick = () => {
        this.visTabButtons.forEach((t) => t.classList.remove("active-vis-tab"));
        tab.classList.add("active-vis-tab");
        const visKey = tab.dataset.vis;
        if (visualizations[visKey]) {
          this.activeVisImage.src = visualizations[visKey];
          this.visCaption.textContent = captions[visKey] || "";
        }
      };
    });
  }

  initResultsActions() {
    this.btnDownloadReport.addEventListener("click", () => {
      if (!this.currentAnalysisData) return;
      const reportUrl = this.currentAnalysisData.visualizations.annotated_report;
      const link = document.createElement("a");
      link.href = reportUrl;
      link.download = `${this.currentAnalysisData.id}_Laboratory_Report.jpg`;
      link.click();
    });

    this.btnExportJson.addEventListener("click", () => {
      if (!this.currentAnalysisData) return;
      window.open(`/api/export/json/${this.currentAnalysisData.id}`, "_blank");
    });

    this.btnAnalyzeAnother.addEventListener("click", () => {
      this.pendingImagePayload = null;
      this.previewCard.style.display = "none";
      this.navigateTo("screenAnalyze");
      if (this.tabCameraMode.classList.contains("active-tab")) {
        this.cameraViewContainer.style.display = "block";
        this.startCameraStream();
      } else {
        this.uploadDropzone.style.display = "block";
      }
    });
  }

  initHistory() {
    this.btnExportCsv.addEventListener("click", () => {
      window.location.href = "/api/export/csv";
    });
  }

  async loadHistory() {
    try {
      const response = await fetch("/api/history");
      const result = await response.json();
      if (!result.success) throw new Error("Could not retrieve registry records.");

      const records = result.data;
      this.historyTableBody.innerHTML = "";

      if (records.length === 0) {
        this.historyTable.style.display = "none";
        this.emptyHistoryBox.style.display = "block";
      } else {
        this.historyTable.style.display = "table";
        this.emptyHistoryBox.style.display = "none";

        records.forEach((rec) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td style="font-family: var(--font-mono); font-weight: 700; color: var(--amber-primary);">${rec.id}</td>
            <td style="color: var(--text-secondary); font-size: 0.8rem;">${rec.timestamp}</td>
            <td style="font-family: var(--font-mono);">${(rec.circularity * 100).toFixed(1)}%</td>
            <td style="font-family: var(--font-mono);">${(rec.burn_ratio * 100).toFixed(1)}%</td>
            <td style="font-family: var(--font-mono); font-weight: 800; color: var(--text-primary);">${rec.perfection_score.toFixed(1)}</td>
            <td><span style="font-weight: 700; font-size: 0.8rem; color: var(--cyan-accent);">${rec.verdict}</span></td>
          `;
          tr.addEventListener("click", () => this.openHistoricalRecord(rec.id));
          this.historyTableBody.appendChild(tr);
        });
      }
    } catch (err) {
      this.showToast(err.message, "error");
    }
  }

  async openHistoricalRecord(analysisId) {
    try {
      const response = await fetch(`/api/analysis/${analysisId}`);
      const result = await response.json();
      if (!result.success) throw new Error(result.error || "Failed to load record.");

      // Reconstruct visualization URLs
      const fullData = result.data;
      const baseFilename = fullData.image_filename.replace("_orig.jpg", "");
      fullData.visualizations = {
        original: `/data/images/${fullData.image_filename}`,
        ideal_overlay: `/data/images/${baseFilename}_ideal.jpg`,
        browning_heatmap: `/data/images/${baseFilename}_browning.jpg`,
        radial_profile: `/data/images/${baseFilename}_profile.png`,
        annotated_report: `/data/images/${baseFilename}_report.jpg`,
      };

      this.displayResults(fullData);
      this.navigateTo("screenResults");
    } catch (err) {
      this.showToast(err.message, "error");
    }
  }

  showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast-msg ${type}`;
    toast.textContent = message;
    this.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(12px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }
}

// Bootstrap application on DOM load
window.addEventListener("DOMContentLoaded", () => {
  window.chapatiApp = new ChapatiApp();
});
