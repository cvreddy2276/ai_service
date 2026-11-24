import React, { useState } from 'react';
import {
  Layers,
  Eye,
  EyeOff,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Sliders,
  Sparkles,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

export default function SliceViewer({
  keySlices = [],
  peakSliceIndex = 36,
  totalSlices = 72,
}) {
  const [selectedSliceIdx, setSelectedSliceIdx] = useState(0);
  const [showGradCam, setShowGradCam] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(65);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(105);

  const slices = keySlices;

  const currentSlice = slices[selectedSliceIdx] || slices[0];

  if (!slices.length) {
    return (
      <div className="glass-panel rounded-2xl border border-slate-800 p-8 text-center text-sm text-slate-400">
        No scan images or attribution maps were saved for this result.
      </div>
    );
  }

  const handleNext = () => {
    setSelectedSliceIdx((prev) => (prev + 1) % slices.length);
  };

  const handlePrev = () => {
    setSelectedSliceIdx((prev) => (prev - 1 + slices.length) % slices.length);
  };

  return (
    <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden shadow-2xl">
      {/* Header Bar */}
      <div className="px-5 py-3.5 bg-slate-900/90 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <Layers className="w-5 h-5 text-brand-400" />
          <h3 className="font-semibold text-white text-sm sm:text-base flex items-center gap-2">
            Axial MRI Slices and Attribution
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
              AXIAL
            </span>
          </h3>
        </div>

      </div>

      {/* Main Viewer Body */}
      <div className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* Slice Image Canvas Container */}
        <div className="lg:col-span-8 flex flex-col items-center justify-center bg-black/80 rounded-xl p-4 border border-slate-800 relative group min-h-[380px]">
          {/* Top Overlay Badges */}
          <div className="absolute top-4 left-4 z-10 flex flex-col gap-1 text-[11px] font-mono text-cyan-400/90 bg-slate-950/80 backdrop-blur-md px-2.5 py-1.5 rounded-lg border border-slate-800">
            <div>Slice: #{currentSlice?.sliceIndex ?? peakSliceIndex} / {totalSlices}</div>
            <div>Plane: AXIAL</div>
            <div>Spacing: 1.0 × 1.0 × 1.0 mm³</div>
          </div>

          {currentSlice?.isPeakAttribution && (
            <div className="absolute top-4 right-4 z-10 flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-500/40 text-cyan-300">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Highest attribution</span>
            </div>
          )}

          {/* Render Slice Image */}
          <div className="relative overflow-hidden rounded-lg flex items-center justify-center max-w-[360px] max-h-[360px]">
            {currentSlice?.imageBase64 ? (
              <>
                <img
                  src={currentSlice.imageBase64}
                  alt={`MRI Slice ${currentSlice.sliceIndex}`}
                  className="w-full h-full object-contain transition-transform"
                  style={{
                    transform: `scale(${zoomLevel})`,
                    filter: `brightness(${brightness}%) contrast(${contrast}%)`,
                  }}
                />
                {showGradCam && currentSlice.gradcamBase64 && (
                  <img
                    src={currentSlice.gradcamBase64}
                    alt={`GradCAM Overlay ${currentSlice.sliceIndex}`}
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity"
                    style={{
                      opacity: heatmapOpacity / 100,
                      transform: `scale(${zoomLevel})`,
                    }}
                  />
                )}
              </>
            ) : (
              /* No patient slice data is available for this output. */
              <div
                className="w-72 h-72 sm:w-80 sm:h-80 rounded-lg relative flex items-center justify-center overflow-hidden border border-slate-800 shadow-inner"
                style={{
                  background: 'radial-gradient(ellipse at center, #1e293b 0%, #0f172a 60%, #020617 100%)',
                  transform: `scale(${zoomLevel})`,
                  filter: `brightness(${brightness}%) contrast(${contrast}%)`,
                }}
              >
                <div className="text-center p-4 z-10 pointer-events-none">
                  <div className="text-xs font-mono text-slate-300">Scan slice unavailable</div>
                  <div className="text-[11px] text-slate-400 mt-1">No image or attribution data was returned.</div>
                </div>
              </div>
            )}
          </div>

          {/* Prev/Next Quick Navigation Overlay */}
          <div className="flex items-center justify-between w-full mt-4 px-2">
            <button
              onClick={handlePrev}
              className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-xs flex items-center gap-1 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" /> Prev Key Slice
            </button>
            <div className="text-xs text-slate-400 font-mono">
              Key Focus {selectedSliceIdx + 1} of {slices.length}
            </div>
            <button
              onClick={handleNext}
              className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-xs flex items-center gap-1 transition-colors"
            >
              Next Key Slice <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Right-Hand Controls & Quantifications */}
        <div className="lg:col-span-4 space-y-5">
          {/* Grad-CAM Heatmap Toggle & Opacity */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                Grad-CAM Activation
              </span>
              <button
                onClick={() => setShowGradCam(!showGradCam)}
                className={`p-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
                  showGradCam
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'bg-slate-800 text-slate-400'
                }`}
              >
                {showGradCam ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                <span>{showGradCam ? 'Visible' : 'Hidden'}</span>
              </button>
            </div>

            {showGradCam && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Overlay Opacity:</span>
                  <span className="font-mono text-cyan-400">{heatmapOpacity}%</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>
            )}
          </div>

          {/* Windowing & Zoom Controls */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3">
            <div className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-brand-400" />
              Window Level & Contrast
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-[11px] text-slate-400">
                <span>Brightness:</span>
                <span className="font-mono">{brightness}%</span>
              </div>
              <input
                type="range"
                min="50"
                max="160"
                value={brightness}
                onChange={(e) => setBrightness(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />

              <div className="flex justify-between text-[11px] text-slate-400 pt-1">
                <span>Contrast:</span>
                <span className="font-mono">{contrast}%</span>
              </div>
              <input
                type="range"
                min="60"
                max="180"
                value={contrast}
                onChange={(e) => setContrast(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-slate-800">
              <span className="text-[11px] text-slate-400">Zoom: {zoomLevel.toFixed(1)}x</span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setZoomLevel((z) => Math.max(0.8, z - 0.2))}
                  className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setZoomLevel(1)}
                  className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px]"
                >
                  Reset
                </button>
                <button
                  onClick={() => setZoomLevel((z) => Math.min(2.0, z + 0.2))}
                  className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>

          {/* Volumetric Quantification Card */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-800 space-y-2">
            <div className="text-xs font-semibold text-slate-300 uppercase tracking-wide">
              Attribution Summary
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-slate-400">Tumor volume:</span>
              <span className="text-sm font-semibold text-slate-300">Not measured by this model</span>
            </div>
            <div className="flex items-baseline justify-between text-xs">
              <span className="text-slate-400">Highest attribution slice:</span>
              <span className="font-mono text-slate-300">Slice #{peakSliceIndex}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
