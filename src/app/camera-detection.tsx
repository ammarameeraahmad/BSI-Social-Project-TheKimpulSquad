'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import * as tf from '@tensorflow/tfjs';
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  type NormalizedLandmark,
  type Category,
} from '@mediapipe/tasks-vision';
import { type Database } from 'firebase/database';
import {
  connectPersonalFirebase,
  getFirebaseDatabase,
  getPersonalFirebaseDatabase,
  publishDatabaseToCatalog,
  unpublishDatabaseFromCatalog,
  type PersonalFirebaseConfig,
  type CatalogDatabaseInput,
} from '../lib/firebase';
import { useAuth } from '../lib/auth-context';

const FRAME_COUNT = 30;
const FEATURE_VERSION = 3;
const FRAMES_ENCODING = 'deflate-base64-v1';
const FACE_LANDMARK_INDICES = [
  // lips / mouth
  61, 291, 13, 14, 78, 308, 82, 312,
  // eyebrows
  46, 53, 65, 276, 283, 295,
  // eyes / eyelids
  33, 263, 159, 145, 386, 374,
  // nose
  1, 4, 6, 19, 24,
  // head pose / jaw
  10, 152, 234, 454, 127, 356,
] as const satisfies number[];
const FACE_POINT_COUNT = FACE_LANDMARK_INDICES.length;
const HAND_POINT_COUNT = 21;
const HAND_COUNT = 2;
const FEATURE_COUNT = (FACE_POINT_COUNT + HAND_POINT_COUNT * HAND_COUNT) * 3 + 3;
const MAX_EPOCHS = 40;
const MIN_RECOGNITION_SCORE = 0.65;
const MIN_RECOGNITION_MARGIN = 0.15;
const MODEL_KEY = 'indexeddb://gesture-lstm-v3';
const LABELS_KEY = 'gesture-lstm-labels-v3';
const REPORT_KEY = 'gesture-lstm-report-v3';
const MODEL_SAMPLES_KEY = 'gesture-lstm-samples-fingerprint-v3';
const SAMPLES_KEY = 'gesture-lab-samples-cache-v1';
const LEGACY_MODEL_KEYS = ['indexeddb://gesture-lstm-v1', 'indexeddb://gesture-lstm-v2'];
const LEGACY_LABEL_KEYS = ['gesture-lstm-labels-v1', 'gesture-lstm-labels-v2'];

type Sample = {
  id: string;
  label: string;
  frames: number[][];
  createdAt: number;
  featureVersion?: number;
};

type StoredSampleRecord = Omit<Sample, 'frames'> & {
  frames: number[][] | string;
  framesEncoding?: string;
};

type TrainingReport = {
  accuracy: number;
  evaluatedSamples: number;
  labelCount: number;
  evaluationSource?: 'test' | 'training';
  perLabel: {
    label: string;
    correct: number;
    total: number;
    confusedWith: string | null;
  }[];
  evaluatedAt: number;
};

type NormalizedHand = {
  points: { x: number; y: number; z: number }[];
  anchor: { x: number; y: number; z: number };
  size: number;
};

function isTrainableSample(sample: Sample) {
  return sample.featureVersion === FEATURE_VERSION &&
    typeof sample.label === 'string' &&
    Array.isArray(sample.frames) &&
    sample.frames.length > 0 &&
    sample.frames.length <= FRAME_COUNT &&
    sample.frames.every((frame) => (
      Array.isArray(frame) &&
      frame.length === FEATURE_COUNT &&
      frame.every((value) => typeof value === 'number' && Number.isFinite(value))
    ));
}

function isFrameArray(value: unknown): value is number[][] {
    return Array.isArray(value) && value.every((frame) => (
      Array.isArray(frame) &&
      frame.every((item) => typeof item === 'number' && Number.isFinite(item))
    ));
}

async function transformBytes(
    bytes: Uint8Array,
    createStream: () => CompressionStream | DecompressionStream
) {
    const stream = createStream();
    const result = new Response(stream.readable).arrayBuffer();
    const writer = stream.writable.getWriter();
    const input = new Uint8Array(new ArrayBuffer(bytes.byteLength));
    input.set(bytes);
    await writer.write(input);
    await writer.close();
    return new Uint8Array(await result);
}

function bytesToBase64(bytes: Uint8Array) {
    let binary = '';
    const chunkSize = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
    }
    return btoa(binary);
}

async function compressFrames(frames: number[][]) {
    const encodedFrames = new TextEncoder().encode(JSON.stringify(frames));
    const compressed = await transformBytes(
      encodedFrames,
      () => new CompressionStream('deflate')
    );
    return bytesToBase64(compressed);
}

async function decompressFrames(encodedFrames: string) {
    const binary = atob(encodedFrames);
    const compressed = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    const decompressed = await transformBytes(
      compressed,
      () => new DecompressionStream('deflate')
    );
    const frames: unknown = JSON.parse(new TextDecoder().decode(decompressed));
    if (!isFrameArray(frames)) {
      throw new Error('Data frame sampel terkompresi tidak valid.');
    }
    return frames;
}

function fingerprintSamples(samples: Sample[]) {
  let hash = 2166136261;
  const sortedSamples = [...samples].sort((left, right) => left.id.localeCompare(right.id));
  for (const sample of sortedSamples) {
    const serialized = JSON.stringify([
      sample.id,
      sample.label,
      sample.createdAt,
      sample.featureVersion,
      sample.frames,
    ]);
    for (let index = 0; index < serialized.length; index += 1) {
      hash ^= serialized.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
  }
  return `${sortedSamples.length}:${(hash >>> 0).toString(16)}`;
}

function formatSampleTimestamp(timestamp: number) {
  const date = new Date(timestamp);
  if (!Number.isFinite(date.getTime())) return 'Waktu tidak diketahui';
  return new Intl.DateTimeFormat('id-ID', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
}

function isSampleRecord(value: unknown): value is StoredSampleRecord {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as Record<string, unknown>;
  const hasValidMetadata = typeof record.id === 'string' &&
    typeof record.label === 'string' &&
    typeof record.createdAt === 'number' &&
    (record.featureVersion === undefined || typeof record.featureVersion === 'number');
  if (!hasValidMetadata) return false;

  if (record.framesEncoding === FRAMES_ENCODING) {
    return typeof record.frames === 'string';
  }
  return record.framesEncoding === undefined && isFrameArray(record.frames);
}

async function decodeSampleRecord(record: StoredSampleRecord): Promise<Sample> {
  let frames: number[][];
  if (record.framesEncoding === FRAMES_ENCODING && typeof record.frames === 'string') {
    frames = await decompressFrames(record.frames);
  } else if (Array.isArray(record.frames)) {
    frames = record.frames;
  } else {
    throw new Error('Format data sampel Firebase tidak didukung.');
  }
  return {
    id: record.id,
    label: record.label,
    frames,
    createdAt: record.createdAt,
    featureVersion: record.featureVersion,
  };
}

async function loadSamplesFromCache(): Promise<Sample[]> {
  try {
    const raw = localStorage.getItem(SAMPLES_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const records = parsed.filter((item): item is StoredSampleRecord => isSampleRecord(item));
    return Promise.all(records.map(decodeSampleRecord));
  } catch (err) {
    console.error('Could not read cached samples:', err);
    return [];
  }
}

function saveSamplesToCache(samples: Sample[]) {
  localStorage.setItem(SAMPLES_KEY, JSON.stringify(samples));
}

async function storeSampleToCache(sample: Sample) {
  const current = await loadSamplesFromCache();
  const next = [sample, ...current.filter((item) => item.id !== sample.id)];
  saveSamplesToCache(next);
}

async function removeSampleFromCache(id: string) {
  const current = await loadSamplesFromCache();
  const next = current.filter((item) => item.id !== id);
  saveSamplesToCache(next);
}

function normalizeHandLandmarks(
  landmarks: NormalizedLandmark[] | undefined,
  width: number,
  height: number
): NormalizedHand | undefined {
  if (!landmarks || landmarks.length === 0) return undefined;

  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  if (!wrist || !middleMcp) return undefined;

  const palmScale = Math.max(
    Math.hypot((middleMcp.x - wrist.x) * width, (middleMcp.y - wrist.y) * height),
    1
  );

  return {
    points: landmarks.map((point) => ({
      x: ((point.x - wrist.x) * width) / palmScale,
      y: ((point.y - wrist.y) * height) / palmScale,
      z: ((point.z - wrist.z) * width) / palmScale,
    })),
    anchor: { x: wrist.x, y: wrist.y, z: wrist.z },
    size: palmScale,
  };
}

function shuffle<T>(items: T[]) {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [items[index], items[swapIndex]] = [items[swapIndex], items[index]];
  }
  return items;
}

function createGestureModel(labelCount: number) {
  const isSingleLabel = labelCount === 1;
  const model = tf.sequential();
  model.add(tf.layers.lstm({ units: 32, inputShape: [FRAME_COUNT, FEATURE_COUNT] }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({
    units: labelCount,
    activation: isSingleLabel ? 'sigmoid' : 'softmax',
  }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: isSingleLabel ? 'binaryCrossentropy' : 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

function createTrainingTensors(
  trainingSamples: Sample[],
  classIndex: Map<string, number>,
  labelCount: number
) {
  const inputValues = trainingSamples.flatMap((sample) => (
    resampleFrames(sample.frames).flat()
  ));
  const targetValues = trainingSamples.flatMap((sample) => {
    const target = new Array(labelCount).fill(0);
    target[classIndex.get(sample.label)!] = 1;
    return target;
  });

  return {
    inputs: tf.tensor3d(inputValues, [trainingSamples.length, FRAME_COUNT, FEATURE_COUNT]),
    targets: tf.tensor2d(targetValues, [trainingSamples.length, labelCount]),
  };
}

function resampleFrames(frames: number[][]) {
  if (frames.length === FRAME_COUNT) return frames;
  return Array.from({ length: FRAME_COUNT }, (_, index) => {
    const sourceIndex = Math.round((index * (frames.length - 1)) / (FRAME_COUNT - 1));
    return frames[sourceIndex];
  });
}

function balanceSamplesByLabel(samples: Map<string, Sample[]>, labels: string[]) {
  const samplesPerLabel = Math.min(
    ...labels.map((classLabel) => samples.get(classLabel)?.length ?? 0)
  );
  return labels.flatMap((classLabel) => (
    shuffle([...(samples.get(classLabel) ?? [])]).slice(0, samplesPerLabel)
  ));
}

function orderHandsBySide(
  hands: NormalizedLandmark[][],
  handedness: Category[][],
  faceCenterX: number
) {
  const candidates = hands.map((landmarks, index) => {
    const category = handedness[index]?.[0]?.categoryName?.toLowerCase();
    // MediaPipe handedness assumes mirrored selfie input; the video fed here is unmirrored.
    const label = category === 'left'
      ? 'right'
      : category === 'right'
        ? 'left'
        : undefined;
    return { landmarks, label };
  });

  if (
    candidates.length === 2 &&
    (!candidates[0].label || !candidates[1].label || candidates[0].label === candidates[1].label)
  ) {
    candidates.sort((left, right) => (
      (left.landmarks[0]?.x ?? 0) - (right.landmarks[0]?.x ?? 0)
    ));
    candidates[0].label = 'right';
    candidates[1].label = 'left';
  } else if (candidates.length === 1 && !candidates[0].label) {
    candidates[0].label = (candidates[0].landmarks[0]?.x ?? faceCenterX) >= faceCenterX
      ? 'left'
      : 'right';
  }

  return {
    left: candidates.find((candidate) => candidate.label === 'left')?.landmarks,
    right: candidates.find((candidate) => candidate.label === 'right')?.landmarks,
  };
}

function makeFeatureFrame(
  face: NormalizedLandmark[] | undefined,
  hands: NormalizedLandmark[][],
  handedness: Category[][],
  width: number,
  height: number
) {
  const features: number[] = [];
  const faceAnchor = face?.[1] ?? { x: 0.5, y: 0.5, z: 0 };
  const faceXs = FACE_LANDMARK_INDICES
    .map((index) => face?.[index]?.x)
    .filter((x): x is number => typeof x === 'number');
  const faceWidth = faceXs.length > 1
    ? Math.max((Math.max(...faceXs) - Math.min(...faceXs)) * width, 1)
    : Math.max(width * 0.25, 1);

  for (const faceIndex of FACE_LANDMARK_INDICES) {
    const point = face?.[faceIndex];
    features.push(
      point ? ((point.x - faceAnchor.x) * width) / faceWidth : 0,
      point ? ((point.y - faceAnchor.y) * height) / faceWidth : 0,
      point ? ((point.z - faceAnchor.z) * width) / faceWidth : 0
    );
  }

  const orderedHands = orderHandsBySide(hands, handedness, faceAnchor.x);
  const leftHand = orderedHands.left;
  const rightHand = orderedHands.right;
  const handBySide = [leftHand, rightHand];

  for (let handIndex = 0; handIndex < HAND_COUNT; handIndex += 1) {
    const normalized = normalizeHandLandmarks(handBySide[handIndex], width, height);
    if (!normalized) {
      for (let pointIndex = 0; pointIndex < HAND_POINT_COUNT; pointIndex += 1) {
        features.push(0, 0, 0);
      }
      continue;
    }
    for (const point of normalized.points) {
      features.push(
        ((point.x * normalized.size) + (normalized.anchor.x - faceAnchor.x) * width) / faceWidth,
        ((point.y * normalized.size) + (normalized.anchor.y - faceAnchor.y) * height) / faceWidth,
        ((point.z * normalized.size) + (normalized.anchor.z - faceAnchor.z) * width) / faceWidth
      );
    }
  }
  features.push(face ? 1 : 0, leftHand ? 1 : 0, rightHand ? 1 : 0);
  return features;
}

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const FACE_CONNECTIONS: [number, number][] = [
  [0, 4], [4, 6], [6, 2], [2, 1], [1, 7], [7, 5], [5, 3], [3, 0],
  [8, 9], [9, 10], [11, 12], [12, 13],
  [14, 16], [16, 17], [17, 15], [18, 19],
  [20, 21], [21, 22], [23, 24], [24, 25],
  [26, 27], [27, 28], [28, 29], [29, 26],
];

function renderFramePreview(frame: number[]) {
  const hasFace = frame[FEATURE_COUNT - 3] === 1;
  const hasLeftHand = frame[FEATURE_COUNT - 2] === 1;
  const hasRightHand = frame[FEATURE_COUNT - 1] === 1;
  const toSvgPoint = (x: number, y: number) => ({
    x: 50 + x * 26,
    y: 38 + y * 25,
  });
  const facePoints = Array.from({ length: FACE_POINT_COUNT }, (_, index) => (
    toSvgPoint(frame[index * 3], frame[index * 3 + 1])
  ));
  const handOffset = FACE_POINT_COUNT * 3;
  const handPoints = Array.from({ length: HAND_COUNT }, (_, handIndex) => (
    Array.from({ length: HAND_POINT_COUNT }, (_, pointIndex) => {
      const pointOffset = handOffset + (handIndex * HAND_POINT_COUNT + pointIndex) * 3;
      return toSvgPoint(frame[pointOffset], frame[pointOffset + 1]);
    })
  ));

  return (
    <svg className="frame-preview" viewBox="0 0 100 76" aria-hidden="true">
      {hasFace && FACE_CONNECTIONS.map(([start, end]) => (
        <line
          className="preview-face-line"
          key={`face-${start}-${end}`}
          x1={facePoints[start].x}
          y1={facePoints[start].y}
          x2={facePoints[end].x}
          y2={facePoints[end].y}
        />
      ))}
      {hasFace && facePoints.map((point, index) => (
        <circle className="preview-face-point" key={`face-point-${index}`} cx={point.x} cy={point.y} r="1.5" />
      ))}
      {handPoints.map((points, handIndex) => {
        const hasHand = handIndex === 0 ? hasLeftHand : hasRightHand;
        if (!hasHand) return null;
        const colorClass = handIndex === 0 ? 'preview-left-hand' : 'preview-right-hand';
        return (
          <g className={colorClass} key={`hand-${handIndex}`}>
            {HAND_CONNECTIONS.map(([start, end]) => (
              <line
                key={`hand-line-${start}-${end}`}
                x1={points[start].x}
                y1={points[start].y}
                x2={points[end].x}
                y2={points[end].y}
              />
            ))}
            {points.map((point, index) => (
              <circle key={`hand-point-${index}`} cx={point.x} cy={point.y} r="1.4" />
            ))}
          </g>
        );
      })}
      {!hasFace && !hasLeftHand && !hasRightHand && (
        <text className="preview-empty" x="50" y="40" textAnchor="middle">Kosong</text>
      )}
    </svg>
  );
}

export default function CameraDetection() {
  const {
    user,
    signOut,
    databaseMode,
    personalFirebaseConfig,
    databasePreferenceLoading,
    setDatabasePreference,
  } = useAuth();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState('Aktifkan kamera untuk mulai.');
  const [isInitializing, setIsInitializing] = useState(false);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [label, setLabel] = useState('Halo');
  const [sampleSearch, setSampleSearch] = useState('');
  const [editingSampleId, setEditingSampleId] = useState<string | null>(null);
  const [editingSampleLabel, setEditingSampleLabel] = useState('');
  const [editingFramesSampleId, setEditingFramesSampleId] = useState<string | null>(null);
  const [selectedFrameIndices, setSelectedFrameIndices] = useState<Set<number>>(new Set());
  const [recordingFrames, setRecordingFrames] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingMessage, setTrainingMessage] = useState('');
  const [databaseMessage, setDatabaseMessage] = useState('Menghubungkan ke Firebase...');
  const [activeDatabase, setActiveDatabase] = useState<Database | null>(null);
  const [isDatabaseSettingsOpen, setIsDatabaseSettingsOpen] = useState(false);
  const [settingsMode, setSettingsMode] = useState<'shared' | 'personal'>('shared');
  const [configDraft, setConfigDraft] = useState<PersonalFirebaseConfig>({
    apiKey: '',
    authDomain: '',
    appId: '',
    projectId: '',
    databaseURL: '',
  });
  const [settingsError, setSettingsError] = useState('');
  const [isConnectingPersonalDatabase, setIsConnectingPersonalDatabase] = useState(false);
  const [isPublishingCatalog, setIsPublishingCatalog] = useState(false);
  const [catalogPublishError, setCatalogPublishError] = useState('');
  const [catalogDraft, setCatalogDraft] = useState<{
    displayName: string;
    description: string;
    labelCount: number;
    sampleCount: number;
  }>({
    displayName: '',
    description: '',
    labelCount: 0,
    sampleCount: 0,
  });
  const [isUnpublishingCatalog, setIsUnpublishingCatalog] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [predictionScores, setPredictionScores] = useState<{ label: string; score: number }[]>([]);
  const [trainingReport, setTrainingReport] = useState<TrainingReport | null>(null);
  const [countdown, setCountdown] = useState(0);
  const [countdownDuration, setCountdownDuration] = useState(3);
  const [recordDuration, setRecordDuration] = useState(3);
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [importError, setImportError] = useState('');
  const [importResult, setImportResult] = useState<{
    successCount: number;
    skippedCount: number;
    errorCount: number;
    errors: string[];
    importedLabels: string[];
  } | null>(null);
  const [isImportDialogOpen, setIsImportDialogOpen] = useState(false);

  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const animationIdRef = useRef<number | null>(null);
  const captureRef = useRef<{ label: string; frames: number[][]; lastFrameAt: number } | null>(null);
  const historyRef = useRef<number[][]>([]);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const labelsRef = useRef<string[]>([]);
  const samplesFingerprintRef = useRef<string | null>(null);
  const inferenceBusyRef = useRef(false);
  const lastInferenceAtRef = useRef(0);
  const frameIntervalRef = useRef(100);
  const countdownRef = useRef(0);
  const startCameraRef = useRef<(() => void) | null>(null);
  const cameraStartingRef = useRef(false);
  const databaseRef = useRef<Database | null>(null);

  useEffect(() => {
    if (databasePreferenceLoading) return;
    if (databaseMode === 'personal' && !personalFirebaseConfig) {
      databaseRef.current = null;
      setActiveDatabase(null);
      return;
    }
    try {
      const database = databaseMode === 'personal'
        ? getPersonalFirebaseDatabase(personalFirebaseConfig!)
        : getFirebaseDatabase();
      databaseRef.current = database;
      setActiveDatabase(database);
    } catch (err) {
      databaseRef.current = null;
      setActiveDatabase(null);
    }
  }, [databaseMode, databasePreferenceLoading, personalFirebaseConfig]);

  useEffect(() => {
    let disposed = false;
    let unsubscribeSamples: (() => void) | null = null;
    async function loadTrainingData() {
      if (databasePreferenceLoading) return;
      const cached = await loadSamplesFromCache();
      if (!disposed) {
        setSamples(cached.sort((left, right) => right.createdAt - left.createdAt));
        samplesFingerprintRef.current = fingerprintSamples(cached);
        setDatabaseMessage('');
      }

      try {
        await tf.ready();
        try {
          const availableModels = await tf.io.listModels();
          for (const legacyModelKey of LEGACY_MODEL_KEYS) {
            if (availableModels[legacyModelKey]) {
              await tf.io.removeModel(legacyModelKey);
            }
          }
          LEGACY_LABEL_KEYS.forEach((key) => localStorage.removeItem(key));
        } catch (err) {
          console.warn('Could not remove the incompatible previous model:', err);
        }
        const storedLabels = localStorage.getItem(LABELS_KEY);
        if (!storedLabels) return;
        const model = await tf.loadLayersModel(MODEL_KEY);
        if (disposed) {
          model.dispose();
          return;
        }
        const labels = JSON.parse(storedLabels);
        if (!Array.isArray(labels) || labels.some((item) => typeof item !== 'string')) {
          model.dispose();
          throw new Error('Daftar label model tersimpan tidak valid.');
        }
        labelsRef.current = labels;
        modelRef.current = model;
        const currentSamplesFingerprint = samplesFingerprintRef.current;
        if (
          currentSamplesFingerprint !== null &&
          localStorage.getItem(MODEL_SAMPLES_KEY) !== currentSamplesFingerprint
        ) {
          await invalidateTrainedModel();
          if (!disposed) setTrainingMessage('Data bersama berubah. Latih ulang model di perangkat ini.');
          return;
        }
        const savedReport = localStorage.getItem(REPORT_KEY);
        if (savedReport) {
          try {
            const parsedReport = JSON.parse(savedReport) as TrainingReport;
            if (
              Number.isFinite(parsedReport.accuracy) &&
              Number.isFinite(parsedReport.evaluatedSamples) &&
              Number.isFinite(parsedReport.labelCount) &&
              Array.isArray(parsedReport.perLabel)
            ) {
              setTrainingReport(parsedReport);
              setTrainingMessage(`Model tersimpan · uji rekaman ${Math.round(parsedReport.accuracy * 100)}%.`);
            } else {
              setTrainingMessage('Model tersimpan dan siap mengenali gestur.');
            }
          } catch (err) {
            console.error('Could not read saved training report:', err);
            setTrainingMessage('Model tersimpan dan siap mengenali gestur.');
          }
        } else {
          setTrainingMessage('Model tersimpan dan siap mengenali gestur.');
        }
        setModelReady(true);
      } catch (err) {
        console.error('Could not load trained gesture model:', err);
        if (!disposed) setTrainingMessage('Model tersimpan tidak dapat dibuka. Latih ulang model.');
      }
    }
    void loadTrainingData();
    return () => {
      disposed = true;
    };
  }, [databasePreferenceLoading]);

  async function beginRecording() {
    const cleanLabel = label.trim();
    if (!cleanLabel || recordingFrames > 0 || isTraining || countdown > 0) return;
    setCountdown(countdownDuration);
    countdownRef.current = countdownDuration;
    for (let i = countdownDuration - 1; i > 0; i -= 1) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setCountdown(i);
      countdownRef.current = i;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setCountdown(0);
    countdownRef.current = 0;
    frameIntervalRef.current = Math.round((recordDuration * 1000) / FRAME_COUNT);
    captureRef.current = { label: cleanLabel, frames: [], lastFrameAt: 0 };
    historyRef.current = [];
    setRecordingFrames(0);
  }

  async function trainModel() {
    const compatibleSamples = samples.filter(isTrainableSample);
    const trainingFingerprint = fingerprintSamples(compatibleSamples);
    const samplesByLabel = new Map<string, Sample[]>();
    for (const sample of compatibleSamples) {
      const group = samplesByLabel.get(sample.label) ?? [];
      group.push(sample);
      samplesByLabel.set(sample.label, group);
    }
    const trainableLabels = [...samplesByLabel.entries()]
      .map(([classLabel]) => classLabel)
      .sort((left, right) => left.localeCompare(right));
    if (trainableLabels.length === 0) {
      setTrainingMessage('Belum ada sampel valid untuk dilatih.');
      return;
    }

    setIsTraining(true);
    setTrainingMessage('Menyiapkan sampel...');
    let validationModel: tf.Sequential | null = null;
    let finalModel: tf.Sequential | null = null;
    const tensors: tf.Tensor[] = [];
    let bestWeights: tf.Tensor[] = [];
    try {
      await tf.ready();
      const classIndex = new Map(trainableLabels.map((classLabel, index) => [classLabel, index]));
      const trainingSamples: Sample[] = [];
      const validationSamples: Sample[] = [];
      const testSamples: Sample[] = [];
      for (const classLabel of trainableLabels) {
        const classSamples = shuffle([...(samplesByLabel.get(classLabel) ?? [])]);
        if (classSamples.length >= 3) {
          testSamples.push(classSamples[0]);
          validationSamples.push(classSamples[1]);
          trainingSamples.push(...classSamples.slice(2));
        } else {
          trainingSamples.push(...classSamples);
        }
      }
      const balancedTrainingSamples = balanceSamplesByLabel(
        new Map(trainableLabels.map((classLabel) => [
          classLabel,
          trainingSamples.filter((sample) => sample.label === classLabel),
        ])),
        trainableLabels
      );

      const trainingTensors = createTrainingTensors(
        balancedTrainingSamples,
        classIndex,
        trainableLabels.length
      );
      tensors.push(trainingTensors.inputs, trainingTensors.targets);

      validationModel = createGestureModel(trainableLabels.length);
      let bestValidationLoss = Number.POSITIVE_INFINITY;
      let bestEpochs = 0;
      let epochsWithoutImprovement = 0;
      const validationTensors = validationSamples.length > 0
        ? createTrainingTensors(validationSamples, classIndex, trainableLabels.length)
        : null;
      if (validationTensors) tensors.push(validationTensors.inputs, validationTensors.targets);
      await validationModel.fit(trainingTensors.inputs, trainingTensors.targets, {
        ...(validationTensors
          ? { validationData: [validationTensors.inputs, validationTensors.targets] as [tf.Tensor, tf.Tensor] }
          : {}),
        epochs: MAX_EPOCHS,
        batchSize: Math.min(8, balancedTrainingSamples.length),
        shuffle: true,
        callbacks: [{
          onEpochEnd: async (epoch, logs) => {
            const validationLoss = Number(logs?.val_loss);
            if (
              validationTensors &&
              Number.isFinite(validationLoss) &&
              validationLoss < bestValidationLoss - 0.001
            ) {
              bestValidationLoss = validationLoss;
              bestEpochs = epoch + 1;
              epochsWithoutImprovement = 0;
              bestWeights.forEach((weight) => weight.dispose());
              bestWeights = validationModel!.getWeights().map((weight) => weight.clone());
            } else if (validationTensors) {
              epochsWithoutImprovement += 1;
            }
            setTrainingMessage(`Melatih ${epoch + 1}/${MAX_EPOCHS}...`);
            if (validationTensors && epochsWithoutImprovement >= 6) {
              validationModel!.stopTraining = true;
            }
          },
        }],
      });

      if (!validationTensors) bestEpochs = MAX_EPOCHS;
      if (bestEpochs === 0) {
        throw new Error('Evaluasi validasi tidak menghasilkan metrik yang valid.');
      }
      if (bestWeights.length > 0) {
        validationModel.setWeights(bestWeights);
        bestWeights.forEach((weight) => weight.dispose());
        bestWeights = [];
      }
      validationModel.dispose();
      validationModel = null;

      setTrainingMessage('Melatih model akhir...');
      const finalTrainingSamples = [...trainingSamples, ...validationSamples];
      const finalTrainingByLabel = new Map(trainableLabels.map((classLabel) => [
        classLabel,
        finalTrainingSamples.filter((sample) => sample.label === classLabel),
      ]));
      const balancedAllSamples = balanceSamplesByLabel(finalTrainingByLabel, trainableLabels);
      const fullTrainingTensors = createTrainingTensors(
        balancedAllSamples,
        classIndex,
        trainableLabels.length
      );
      tensors.push(
        fullTrainingTensors.inputs,
        fullTrainingTensors.targets
      );
      finalModel = createGestureModel(trainableLabels.length);
      await finalModel.fit(fullTrainingTensors.inputs, fullTrainingTensors.targets, {
        epochs: bestEpochs,
        batchSize: Math.min(8, balancedAllSamples.length),
        shuffle: true,
        callbacks: [{
          onEpochEnd: (epoch) => {
            setTrainingMessage(`Melatih semua data · ${epoch + 1}/${bestEpochs}...`);
          },
        }],
      });

      const hasTestSampleForEveryLabel = trainableLabels.every((classLabel) => (
        testSamples.some((sample) => sample.label === classLabel)
      ));
      const evaluationSamples = hasTestSampleForEveryLabel
        ? testSamples
        : balancedAllSamples;
      const evaluationTensors = createTrainingTensors(
        evaluationSamples,
        classIndex,
        trainableLabels.length
      );
      tensors.push(evaluationTensors.inputs, evaluationTensors.targets);
      const predictedTensor = finalModel.predict(evaluationTensors.inputs) as tf.Tensor2D;
      tensors.push(predictedTensor);
      const predictedScores = await predictedTensor.array();
      const perLabelCounts = new Map<string, { correct: number; total: number }>();
      const confusionCounts = new Map<string, Map<string, number>>();
      trainableLabels.forEach((classLabel) => {
        perLabelCounts.set(classLabel, { correct: 0, total: 0 });
        confusionCounts.set(classLabel, new Map());
      });
      evaluationSamples.forEach((sample, index) => {
        const scores = predictedScores[index] ?? [];
        const predictedIndex = scores.reduce(
          (bestIndex, score, scoreIndex) => score > (scores[bestIndex] ?? -Infinity)
            ? scoreIndex
            : bestIndex,
          0
        );
        const result = perLabelCounts.get(sample.label)!;
        result.total += 1;
        if (predictedIndex === classIndex.get(sample.label)) {
          result.correct += 1;
        } else {
          const predictedLabel = trainableLabels[predictedIndex];
          const confusions = confusionCounts.get(sample.label)!;
          confusions.set(predictedLabel, (confusions.get(predictedLabel) ?? 0) + 1);
        }
      });
      const perLabel = [...perLabelCounts.entries()].map(([classLabel, result]) => {
        const mostCommonConfusion = [...(confusionCounts.get(classLabel)?.entries() ?? [])]
          .sort((left, right) => right[1] - left[1])[0]?.[0] ?? null;
        return { label: classLabel, ...result, confusedWith: mostCommonConfusion };
      });
      const report: TrainingReport = {
        accuracy: perLabel.reduce(
          (sum, result) => sum + result.correct / result.total,
          0
        ) / perLabel.length,
        evaluatedSamples: evaluationSamples.length,
        labelCount: trainableLabels.length,
        perLabel,
        evaluatedAt: Date.now(),
        evaluationSource: hasTestSampleForEveryLabel ? 'test' : 'training',
      };

      if (
        samplesFingerprintRef.current !== null &&
        samplesFingerprintRef.current !== trainingFingerprint
      ) {
        throw new Error('Data bersama berubah selama pelatihan. Coba latih ulang.');
      }
      await finalModel.save(MODEL_KEY);
      localStorage.setItem(LABELS_KEY, JSON.stringify(trainableLabels));
      localStorage.setItem(REPORT_KEY, JSON.stringify(report));
      localStorage.setItem(MODEL_SAMPLES_KEY, trainingFingerprint);
      modelRef.current?.dispose();
      modelRef.current = finalModel;
      finalModel = null;
      labelsRef.current = trainableLabels;
      setTrainingReport(report);
      setModelReady(true);
      setPredictionScores([]);
      setTrainingMessage(
        `Akurasi: ${Math.round(report.accuracy * 100)}%` +
        (trainableLabels.length === 1 ? ' · hanya mengenali label ini.' : '')
      );
    } catch (err) {
      console.error('LSTM training failed:', err);
      setTrainingMessage(`Pelatihan gagal: ${err instanceof Error ? err.message : 'periksa kapasitas browser'}`);
    } finally {
      tensors.forEach((tensor) => tensor.dispose());
      bestWeights.forEach((weight) => weight.dispose());
      validationModel?.dispose();
      finalModel?.dispose();
      setIsTraining(false);
    }
  }

  useEffect(() => {
    let disposed = false;
    let stream: MediaStream | null = null;
    let faceLandmarker: FaceLandmarker | null = null;
    let handLandmarker: HandLandmarker | null = null;
    let lastHandStatus = '';
    let initStage = 'izin kamera';

    async function init() {
      try {
        if (!window.isSecureContext) {
          throw new Error('Buka aplikasi melalui localhost atau koneksi HTTPS.');
        }
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Browser ini tidak mendukung akses kamera.');
        }

        setIsInitializing(true);
        setStatus('Meminta izin kamera...');
        initStage = 'izin kamera';
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 30 },
          },
        });
        if (disposed) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        const video = videoRef.current;
        if (!video) {
          throw new Error('Elemen video kamera tidak tersedia.');
        }

        video.srcObject = stream;
        initStage = 'menyalakan kamera';
        await video.play();
        if (disposed) return;

        initStage = 'model visi';
        setStatus('Memuat model pengenal...');
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm'
        );
        if (disposed) return;

        initStage = 'face model';
        setStatus('Memuat model wajah...');
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
        });
        if (disposed) {
          faceLandmarker.close();
          return;
        }

        initStage = 'hand model';
        setStatus('Memuat model tangan...');
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.35,
          minHandPresenceConfidence: 0.35,
          minTrackingConfidence: 0.35,
        });
        if (disposed) {
          handLandmarker.close();
          faceLandmarker.close();
          return;
        }

        faceLandmarkerRef.current = faceLandmarker;
        handLandmarkerRef.current = handLandmarker;
        setStatus('Kamera aktif');
        void detect();
      } catch (err) {
        stream?.getTracks().forEach((track) => track.stop());
        if (videoRef.current?.srcObject === stream) {
          videoRef.current.srcObject = null;
        }
        handLandmarker?.close();
        faceLandmarker?.close();
        if (handLandmarkerRef.current === handLandmarker) {
          handLandmarkerRef.current = null;
        }
        if (faceLandmarkerRef.current === faceLandmarker) {
          faceLandmarkerRef.current = null;
        }
        if (disposed || (err instanceof DOMException && err.name === 'AbortError')) {
          return;
        }
        console.error(`Initialization failed during ${initStage}:`, err);
        const message = err instanceof DOMException && err.name === 'NotAllowedError'
          ? 'Izin kamera ditolak. Izinkan kamera untuk localhost lewat ikon di address bar atau Setelan situs Chrome, lalu coba lagi.'
          : err instanceof DOMException && err.name === 'NotFoundError'
            ? 'Kamera tidak ditemukan. Sambungkan kamera lalu coba lagi.'
            : err instanceof DOMException && err.name === 'NotReadableError'
              ? 'Kamera sedang dipakai aplikasi lain. Tutup aplikasi itu lalu coba lagi.'
              : err instanceof Error
                ? err.message
                : 'Periksa izin kamera lalu coba lagi.';
        setStatus(`Kamera gagal: ${message}`);
      } finally {
        cameraStartingRef.current = false;
        if (!disposed) setIsInitializing(false);
      }
    }

    function drawHandLandmarks(
      ctx: CanvasRenderingContext2D,
      landmarks: NormalizedLandmark[],
      width: number,
      height: number,
      color: string
    ) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (const [start, end] of HAND_CONNECTIONS) {
        const p1 = landmarks[start];
        const p2 = landmarks[end];
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }

      ctx.fillStyle = color;
      for (const point of landmarks) {
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    async function detect() {
      if (
        !videoRef.current ||
        !canvasRef.current ||
        !faceLandmarkerRef.current ||
        !handLandmarkerRef.current
      ) {
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      try {
        const timestamp = performance.now();
        const faceResults = faceLandmarkerRef.current.detectForVideo(video, timestamp);
        const hands = handLandmarkerRef.current.detectForVideo(video, timestamp);
        if (disposed) return;

        const face = faceResults.faceLandmarks?.[0];
        const faceCenterX = face?.[1]?.x ?? 0.5;
        const handsBySide = orderHandsBySide(hands.landmarks, hands.handedness, faceCenterX);
        const detectedSides = [
          handsBySide.left ? 'kiri' : '',
          handsBySide.right ? 'kanan' : '',
        ].filter(Boolean).join(', ');
        const handStatus = `Kamera aktif · ${hands.landmarks.length}/2 tangan${detectedSides ? ` · ${detectedSides}` : ''}`;
        if (lastHandStatus !== handStatus) {
          lastHandStatus = handStatus;
          setStatus(handStatus);
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (faceResults.faceLandmarks) {
          for (const landmarks of faceResults.faceLandmarks) {
            ctx.fillStyle = '#4ade80';
            for (const point of landmarks) {
              ctx.beginPath();
              ctx.arc(
                point.x * canvas.width,
                point.y * canvas.height,
                1.25,
                0,
                2 * Math.PI
              );
              ctx.fill();
            }
          }
        }

        if (handsBySide.left) {
          drawHandLandmarks(ctx, handsBySide.left, canvas.width, canvas.height, '#fb7185');
        }
        if (handsBySide.right) {
          drawHandLandmarks(ctx, handsBySide.right, canvas.width, canvas.height, '#facc15');
        }

        if (countdownRef.current > 0) {
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 120px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(countdownRef.current.toString(), canvas.width / 2, canvas.height / 2);
        }

        const featureFrame = makeFeatureFrame(face, hands.landmarks, hands.handedness, canvas.width, canvas.height);
        const capture = captureRef.current;
        if (capture && timestamp - capture.lastFrameAt >= frameIntervalRef.current) {
          capture.frames.push(featureFrame);
          capture.lastFrameAt = timestamp;
          setRecordingFrames(capture.frames.length);
          if (capture.frames.length >= FRAME_COUNT) {
            captureRef.current = null;
            const sample: Sample = {
              id: crypto.randomUUID(),
              label: capture.label,
              frames: capture.frames,
              createdAt: Date.now(),
              featureVersion: FEATURE_VERSION,
            };
            void invalidateTrainedModel().then(() => storeSampleToCache(sample)).then(() => {
              setSamples((current) => [
                sample,
                ...current.filter((item) => item.id !== sample.id),
              ].sort((left, right) => right.createdAt - left.createdAt));
              setRecordingFrames(0);
              setTrainingMessage(`Sampel "${sample.label}" tersimpan di cache browser.`);
            }).catch((err: unknown) => {
              console.error('Could not save sample:', err);
              setRecordingFrames(0);
              setTrainingMessage(`Sampel gagal disimpan: ${err instanceof Error ? err.message : 'kesalahan penyimpanan'}`);
            });
          }
        }

        historyRef.current.push(featureFrame);
        if (historyRef.current.length > FRAME_COUNT) historyRef.current.shift();
        if (
          modelRef.current && historyRef.current.length === FRAME_COUNT &&
          !inferenceBusyRef.current && timestamp - lastInferenceAtRef.current > 700
        ) {
          inferenceBusyRef.current = true;
          lastInferenceAtRef.current = timestamp;
          const input = tf.tensor3d([historyRef.current], [1, FRAME_COUNT, FEATURE_COUNT]);
          const prediction = tf.tidy(() => modelRef.current!.predict(input) as tf.Tensor);
          input.dispose();
          void prediction.data().then((scores) => {
            const rankedScores = Array.from(scores, (score, index) => ({
              label: labelsRef.current[index],
              score: Number(score),
            }))
              .filter((item): item is { label: string; score: number } => (
                typeof item.label === 'string' && Number.isFinite(item.score)
              ))
              .sort((left, right) => right.score - left.score);
            setPredictionScores(rankedScores);
          }).catch((err: unknown) => {
            console.error('Gesture inference failed:', err);
          }).finally(() => {
            prediction.dispose();
            inferenceBusyRef.current = false;
          });
        }

        animationIdRef.current = requestAnimationFrame(() => {
          void detect();
        });
      } catch (err) {
        if (!disposed) {
          setStatus(
            `Detection error: ${err instanceof Error ? err.message : 'Failed to detect hands'}`
          );
        }
      }
    }

    startCameraRef.current = () => {
      if (disposed || cameraStartingRef.current) return;
      cameraStartingRef.current = true;
      void init();
    };

    return () => {
      disposed = true;
      startCameraRef.current = null;
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      stream?.getTracks().forEach((track) => track.stop());
      if (videoRef.current?.srcObject === stream) {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      }
      faceLandmarkerRef.current?.close();
      handLandmarkerRef.current?.close();
      modelRef.current?.dispose();
      faceLandmarkerRef.current = null;
      handLandmarkerRef.current = null;
    };
  }, []);

  const usableSamples = samples.filter(isTrainableSample);
  const labelCounts = usableSamples.reduce<Record<string, number>>((counts, sample) => {
    counts[sample.label] = (counts[sample.label] ?? 0) + 1;
    return counts;
  }, {});
  const samplesByLabel = samples.reduce<Record<string, Sample[]>>((groups, sample) => {
    (groups[sample.label] ??= []).push(sample);
    return groups;
  }, {});
  const normalizedSampleSearch = sampleSearch.trim().toLocaleLowerCase('id-ID');
  const filteredSamplesByLabel = normalizedSampleSearch
    ? Object.fromEntries(
      Object.entries(samplesByLabel).filter(([groupLabel]) => (
        groupLabel.toLocaleLowerCase('id-ID').includes(normalizedSampleSearch)
      ))
    )
    : samplesByLabel;
  const canTrain = usableSamples.length > 0;
  const unsurePrediction = predictionScores.length > 0 && (
    predictionScores[0].score < MIN_RECOGNITION_SCORE ||
    predictionScores[0].score - (predictionScores[1]?.score ?? 0) < MIN_RECOGNITION_MARGIN
  );

  async function invalidateTrainedModel() {
    if ((await tf.io.listModels())[MODEL_KEY]) {
      await tf.io.removeModel(MODEL_KEY);
    }
    localStorage.removeItem(LABELS_KEY);
    localStorage.removeItem(REPORT_KEY);
    localStorage.removeItem(MODEL_SAMPLES_KEY);
    modelRef.current?.dispose();
    modelRef.current = null;
    labelsRef.current = [];
    setModelReady(false);
    setPredictionScores([]);
    setTrainingReport(null);
  }

  async function saveSampleLabel(sample: Sample) {
    const cleanLabel = editingSampleLabel.trim();
    if (!cleanLabel) {
      setTrainingMessage('Nama label tidak boleh kosong.');
      return;
    }

    const updatedSample = { ...sample, label: cleanLabel };
    try {
      await invalidateTrainedModel();
      await storeSampleToCache(updatedSample);
      setSamples((current) => current.map((item) => (
        item.id === sample.id ? updatedSample : item
      )));
      setEditingSampleId(null);
      setTrainingMessage('Data diperbarui. Latih ulang model.');
    } catch (err) {
      console.error('Could not update saved sample:', err);
      setTrainingMessage(`Data gagal diperbarui: ${err instanceof Error ? err.message : 'kesalahan penyimpanan'}`);
    }
  }

  async function deleteSample(sample: Sample) {
    if (!window.confirm(`Hapus data "${sample.label}"?`)) return;
    try {
      await invalidateTrainedModel();
      await removeSampleFromCache(sample.id);
      setSamples((current) => current.filter((item) => item.id !== sample.id));
      setTrainingMessage('Data dihapus. Latih ulang model.');
    } catch (err) {
      console.error('Could not delete saved sample:', err);
      setTrainingMessage(`Data gagal dihapus: ${err instanceof Error ? err.message : 'kesalahan penyimpanan'}`);
    }
  }

  async function saveSampleFrames(sample: Sample) {
    const framesToRemove = selectedFrameIndices;
    if (framesToRemove.size === 0) {
      setEditingFramesSampleId(null);
      return;
    }

    const remainingFrames = sample.frames.filter((_, index) => !framesToRemove.has(index));
    if (remainingFrames.length === 0) {
      setTrainingMessage('Sisakan minimal satu frame dalam rekaman.');
      return;
    }

    const updatedSample = { ...sample, frames: remainingFrames };
    try {
      await invalidateTrainedModel();
      await storeSampleToCache(updatedSample);
      setSamples((current) => current.map((item) => (
        item.id === sample.id ? updatedSample : item
      )));
      setEditingFramesSampleId(null);
      setSelectedFrameIndices(new Set());
      setTrainingMessage(`${framesToRemove.size} frame dihapus. Latih ulang model.`);
    } catch (err) {
      console.error('Could not update saved sample frames:', err);
      setTrainingMessage(`Frame gagal diperbarui: ${err instanceof Error ? err.message : 'kesalahan penyimpanan'}`);
    }
  }

  function openDatabaseSettings() {
    setSettingsMode(databaseMode);
    setConfigDraft(personalFirebaseConfig ?? {
      apiKey: '',
      authDomain: '',
      appId: '',
      projectId: '',
      databaseURL: '',
    });
    setSettingsError('');
    setIsDatabaseSettingsOpen(true);
  }

  async function saveDatabaseSettings() {
    setSettingsError('');
    if (settingsMode === 'shared') {
      setDatabasePreference('shared');
      setIsDatabaseSettingsOpen(false);
      return;
    }

    const config = {
      apiKey: configDraft.apiKey.trim(),
      authDomain: configDraft.authDomain.trim().replace(/^https?:\/\//i, '').replace(/\/+$/, ''),
      appId: configDraft.appId.trim(),
      projectId: configDraft.projectId.trim(),
      databaseURL: configDraft.databaseURL.trim(),
    };
    if (Object.values(config).some((value) => !value)) {
      setSettingsError('Lengkapi semua konfigurasi Firebase.');
      return;
    }

    setIsConnectingPersonalDatabase(true);
    try {
      await connectPersonalFirebase(config);
      setDatabasePreference('personal', config);
      setIsDatabaseSettingsOpen(false);
    } catch (err) {
      setSettingsError(`Koneksi gagal: ${err instanceof Error ? err.message : 'periksa konfigurasi dan izin Firebase.'}`);
    } finally {
      setIsConnectingPersonalDatabase(false);
    }
  }

  async function handlePublishCatalog() {
    if (!activeDatabase || !user || !personalFirebaseConfig) return;
    setCatalogPublishError('');
    if (!catalogDraft.displayName.trim()) {
      setCatalogPublishError('Nama tampilan tidak boleh kosong.');
      return;
    }
    if (usableSamples.length === 0) {
      setCatalogPublishError('Tidak ada sampel valid untuk dipublikasikan.');
      return;
    }
    setIsPublishingCatalog(true);
    try {
      const input: CatalogDatabaseInput = {
        config: personalFirebaseConfig,
        displayName: catalogDraft.displayName.trim(),
        description: catalogDraft.description.trim(),
        labelCount: Object.keys(labelCounts).length,
        sampleCount: usableSamples.length,
      };
      await publishDatabaseToCatalog(
        activeDatabase,
        input,
        user.uid,
        user.displayName,
        user.photoURL
      );
      setCatalogDraft({ displayName: '', description: '', labelCount: 0, sampleCount: 0 });
      setTrainingMessage('Database dipublikasikan ke katalog!');
    } catch (err) {
      setCatalogPublishError(`Publikasi gagal: ${err instanceof Error ? err.message : 'periksa koneksi dan izin Firebase'}`);
    } finally {
      setIsPublishingCatalog(false);
    }
  }

  async function handleExport() {
    if (samples.length === 0) {
      setTrainingMessage('Tidak ada data untuk diekspor.');
      return;
    }
    setIsExporting(true);
    try {
      const exportData = samples.map((sample) => ({
        id: sample.id,
        label: sample.label,
        frames: sample.frames,
        createdAt: sample.createdAt,
        ...(sample.featureVersion === undefined ? {} : { featureVersion: sample.featureVersion }),
      }));
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `gesture-export-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      setTrainingMessage(`Berhasil mengekspor ${samples.length} sampel.`);
    } catch (err) {
      console.error('Export failed:', err);
      setTrainingMessage(`Ekspor gagal: ${err instanceof Error ? err.message : 'kesalahan tidak diketahui'}`);
    } finally {
      setIsExporting(false);
    }
  }

  async function handleImport(file: File) {
    setIsImporting(true);
    setImportError('');
    setImportResult(null);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      if (!Array.isArray(parsed)) {
        throw new Error('Format file JSON tidak valid.');
      }
      const existing = await loadSamplesFromCache();
      const existingLabels = new Set(existing.map((s) => s.label));
      const imported: Sample[] = [];
      const skipped: string[] = [];
      const errors: string[] = [];
      for (const item of parsed) {
        if (!isSampleRecord(item)) {
          errors.push(`Melewati item tidak valid: ${JSON.stringify(item).slice(0, 80)}`);
          continue;
        }
        const sample = await decodeSampleRecord(item);
        if (existingLabels.has(sample.label)) {
          skipped.push(sample.label);
          continue;
        }
        imported.push(sample);
        existingLabels.add(sample.label);
      }
      const next = [...imported, ...existing];
      saveSamplesToCache(next);
      setSamples(next.sort((left, right) => right.createdAt - left.createdAt));
      setImportResult({
        successCount: imported.length,
        skippedCount: skipped.length,
        errorCount: errors.length,
        errors,
        importedLabels: imported.map((s) => s.label),
      });
      if (imported.length > 0) {
        setTrainingMessage(`Impor selesai: ${imported.length} data baru ditambahkan. Latih ulang model.`);
        await invalidateTrainedModel();
      } else if (skipped.length > 0) {
        setTrainingMessage(`Impor selesai: ${skipped.length} data dilewati (label sudah ada).`);
      } else {
        setTrainingMessage('Impor selesai: Tidak ada data baru yang ditambahkan.');
      }
    } catch (err) {
      console.error('Import failed:', err);
      setImportError(`Impor gagal: ${err instanceof Error ? err.message : 'kesalahan tidak diketahui'}`);
    } finally {
      setIsImporting(false);
    }
  }

  function openImportDialog() {
    setImportError('');
    setImportResult(null);
    setIsImportDialogOpen(true);
  }

  function closeImportDialog() {
    setIsImportDialogOpen(false);
    setImportError('');
    setImportResult(null);
  }

  return (
    <main className="studio-shell">
      <div className="studio-layout">
        <section className="camera-column" aria-label="Kamera dan pengenalan gestur">
          <div className="camera-frame">
            <div className="camera-header">
              <div className="camera-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true" style={{width: 18, height: 18}}>
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5l4 4h12a2 2 0 0 1 2 2v10z" />
                </svg>
                <span>Gesture Lab</span>
              </div>
              <div className="user-menu">
                {user && (
                  <>
                    <Link href="/katalog" className="user-button katalog-link" title="Katalog Database" aria-label="Katalog Database">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true" style={{width: 20, height: 20}}>
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                      </svg>
                    </Link>
                    <button className="user-button" onClick={openDatabaseSettings} title="Pengaturan akun" aria-label="Pengaturan akun">
                      {user.photoURL ? (
                        <img src={user.photoURL} alt="" className="user-avatar" />
                      ) : (
                        <span className="user-avatar-placeholder">{user.displayName?.[0]?.toUpperCase() ?? 'U'}</span>
                      )}
                    </button>
                  </>
                )}
              </div>
            </div>
            <video ref={videoRef} className="camera-source" playsInline />
            <canvas ref={canvasRef} className="camera-output" />
            {!status.startsWith('Kamera aktif') && (
              <div className="camera-access" role="status">
                <p>{isInitializing ? status : status.replace(/^Kamera gagal: /, '')}</p>
                {!isInitializing && (
                  <button
                    className="camera-access-button"
                    onClick={() => startCameraRef.current?.()}
                  >
                    {status.startsWith('Kamera gagal:') ? 'Coba Lagi' : 'Izinkan Kamera'}
                  </button>
                )}
              </div>
            )}
            {recordingFrames > 0 && <div className="recording-indicator"><i /> {recordingFrames}/{FRAME_COUNT}</div>}
            {countdown > 0 && <div className="countdown-overlay">{countdown}</div>}
            <div className="prediction-panel" aria-live="polite">
              <div className="prediction-heading">
                <strong>Skor model</strong>
                <span>{unsurePrediction ? 'Belum yakin' : 'prediksi'}</span>
              </div>
              {predictionScores.length > 0 ? (
                <div className="prediction-list">
                  {predictionScores.map(({ label: predictionLabel, score }, index) => (
                    <div className={`prediction-item${index === 0 ? ' is-top' : ''}`} key={predictionLabel}>
                      <div className="prediction-label">
                        <span title={predictionLabel}>{predictionLabel}</span>
                        <strong>{Math.round(score * 100)}%</strong>
                      </div>
                      <div className="prediction-track">
                        <span style={{ width: `${Math.max(0, Math.min(100, score * 100))}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="prediction-empty">
                  {modelReady ? 'Menganalisis gerakan...' : 'Latih model untuk melihat hasil.'}
                </p>
              )}
              {predictionScores.length > 0 && (
                <p className="prediction-disclaimer">Skor bukan persentase akurasi.</p>
              )}
            </div>
          </div>
        </section>

        <aside className="training-panel">
          <label className="field-label" htmlFor="gesture-label">Label gestur</label>
          <input id="gesture-label" className="label-input" value={label} onChange={(event) => setLabel(event.target.value)} maxLength={48} />
          <button className="record-button" onClick={() => void beginRecording()} disabled={!status.startsWith('Kamera aktif') || recordingFrames > 0 || isTraining || countdown > 0 || !label.trim()}>
            <span className="record-icon" /> {recordingFrames > 0 ? `${recordingFrames}/${FRAME_COUNT}` : 'Rekam Sampel'}
          </button>

          <div className="record-settings">
            <label>
              <span>Countdown</span>
              <input type="number" value={countdownDuration} onChange={(event) => setCountdownDuration(Number(event.target.value))} min="0" max="10" />
            </label>
            <label>
              <span>Durasi</span>
              <input type="number" value={recordDuration} onChange={(event) => setRecordDuration(Number(event.target.value))} min="1" max="10" />
            </label>
          </div>

          {samples.length > usableSamples.length && (
            <p className="minimum-note is-warning">
              {samples.length - usableSamples.length} data lama perlu direkam ulang.
            </p>
          )}
          <button className="train-button" onClick={() => void trainModel()} disabled={isTraining || !canTrain}>
            {isTraining ? <><span className="spinner" /> Training...</> : 'Latih Model'}
          </button>
          <p className="training-message">{trainingMessage}</p>
          {trainingReport && (
            <div className="evaluation-note">
              <p>
                Akurasi: {Math.round(trainingReport.accuracy * 100)}% · {trainingReport.evaluatedSamples} sampel
                {trainingReport.labelCount === 1 && ' · hanya 1 label, belum bisa membedakan gestur lain'}
              </p>
              <details>
                <summary>Per label</summary>
                <ul>
                  {trainingReport.perLabel.map((result) => (
                    <li key={result.label}>
                      {result.label}: {result.correct}/{result.total} benar
                      {result.confusedWith ? ` · tertukar dengan ${result.confusedWith}` : ''}
                    </li>
                  ))}
                </ul>
              </details>
            </div>
          )}
        </aside>

        <section className="dataset-panel" aria-labelledby="dataset-title">
          <div className="dataset-heading">
            <h2 id="dataset-title">Data latihan</h2>
            <span>{samples.length} sampel · {usableSamples.length} valid · {Object.keys(labelCounts).length} label</span>
          </div>
          <div className="dataset-actions">
            <button
              className="sample-action"
              onClick={handleExport}
              disabled={isExporting || samples.length === 0}
              aria-label="Ekspor data gestur ke file JSON"
            >
              {isExporting ? <><span className="spinner" style={{width: 14, height: 14}} /> Mengekspor...</> : 'Ekspor Data'}
            </button>
            <button
              className="sample-action"
              onClick={openImportDialog}
              disabled={isImporting || isImportDialogOpen}
              aria-label="Impor data gestur dari file JSON"
            >
              Impor Data
            </button>
          </div>
          <p className="database-message" role="status">{databaseMessage}</p>
          {samples.length > 0 && (
            <input
              className="label-input sample-search"
              type="search"
              value={sampleSearch}
              onChange={(event) => setSampleSearch(event.target.value)}
              placeholder="Cari data berdasarkan label..."
              aria-label="Cari data latihan berdasarkan label"
            />
          )}
          {samples.length === 0 ? (
            <div className="empty-state"><p>Belum ada data. Rekam sampel pertama.</p></div>
          ) : Object.keys(filteredSamplesByLabel).length === 0 ? (
            <div className="empty-state"><p>Tidak ada data dengan label “{sampleSearch.trim()}”.</p></div>
          ) : (
            <div className="sample-list">
              {Object.entries(filteredSamplesByLabel).map(([groupLabel, labelSamples]) => (
                <section className="sample-group" key={groupLabel} aria-label={`Rekaman label ${groupLabel}`}>
                  <div className="sample-group-heading">
                    <strong>{groupLabel}</strong>
                    <span>{labelSamples.length} rekaman</span>
                  </div>
                  {labelSamples.map((sample) => {
                    const sampleIsValid = isTrainableSample(sample);
                    const allExceptFirstSelected =
                      selectedFrameIndices.size === sample.frames.length - 1 &&
                      sample.frames.slice(1).every((_, index) => selectedFrameIndices.has(index + 1));
                return (
                  <div className="sample-entry" key={sample.id}>
                    <div className="sample-row">
                      {editingSampleId === sample.id ? (
                        <>
                          <input
                            className="label-input"
                            value={editingSampleLabel}
                            onChange={(event) => setEditingSampleLabel(event.target.value)}
                            maxLength={48}
                            aria-label={`Nama label untuk data ${sample.label}`}
                          />
                          <span className="sample-status">
                            {sample.frames.length} frame
                          </span>
                          <button
                            className="sample-action"
                            onClick={() => void saveSampleLabel(sample)}
                            disabled={isTraining}
                          >
                            Simpan
                          </button>
                          <button
                            className="sample-action"
                            onClick={() => setEditingSampleId(null)}
                            disabled={isTraining}
                          >
                            Batal
                          </button>
                        </>
                      ) : (
                        <>
                          <time className="sample-date">{formatSampleTimestamp(sample.createdAt)}</time>
                          <span className="sample-status">
                            {sample.frames.length} frame
                          </span>
                          {sampleIsValid && (
                            <button
                              className="sample-action"
                              onClick={() => {
                                setEditingFramesSampleId((current) => (
                                  current === sample.id ? null : sample.id
                                ));
                                setSelectedFrameIndices(new Set());
                              }}
                              disabled={isTraining}
                            >
                              {editingFramesSampleId === sample.id ? 'Tutup frame' : 'Edit frame'}
                            </button>
                          )}
                          <button
                            className="sample-action"
                            onClick={() => {
                              setEditingSampleId(sample.id);
                              setEditingSampleLabel(sample.label);
                            }}
                            disabled={isTraining}
                          >
                            Ubah label
                          </button>
                          <button
                            className="sample-action is-danger"
                            onClick={() => void deleteSample(sample)}
                            disabled={isTraining}
                          >
                            Hapus
                          </button>
                        </>
                      )}
                    </div>
                    {editingFramesSampleId === sample.id && sampleIsValid && (
                      <div className="frame-editor">
                        <p>
                          Pratinjau menunjukkan kerangka landmark, bukan gambar kamera. Klik frame yang ingin dibuang,
                          atau pilih semua sekaligus (frame pertama akan disisakan).
                        </p>
                        <div className="frame-strip" role="group" aria-label={`Pilih frame yang akan dihapus dari ${sample.label}`}>
                          {sample.frames.map((frame, index) => {
                            const isSelected = selectedFrameIndices.has(index);
                            return (
                              <button
                                className={`frame-button${isSelected ? ' is-selected' : ''}`}
                                type="button"
                                key={index}
                                aria-pressed={isSelected}
                                aria-label={`Frame ${index + 1}${isSelected ? ', dipilih untuk dihapus' : ''}`}
                                onClick={() => setSelectedFrameIndices((current) => {
                                  const next = new Set(current);
                                  if (next.has(index)) next.delete(index);
                                  else next.add(index);
                                  return next;
                                })}
                                disabled={isTraining}
                              >
                                {renderFramePreview(frame)}
                                <span>{index + 1}</span>
                              </button>
                            );
                          })}
                        </div>
                        <div className="frame-editor-actions">
                          <span>{sample.frames.length - selectedFrameIndices.size} dari {sample.frames.length} frame tersisa</span>
                          <button
                            className="sample-action"
                            type="button"
                            onClick={() => {
                              setSelectedFrameIndices(allExceptFirstSelected
                                ? new Set()
                                : new Set(sample.frames.slice(1).map((_, index) => index + 1)));
                            }}
                            disabled={isTraining || sample.frames.length <= 1}
                          >
                            {allExceptFirstSelected
                              ? 'Batal pilih semua'
                              : 'Pilih semua (sisakan 1)'}
                          </button>
                          <button
                            className="sample-action is-danger"
                            type="button"
                            onClick={() => void saveSampleFrames(sample)}
                            disabled={isTraining || selectedFrameIndices.size === 0 || selectedFrameIndices.size >= sample.frames.length}
                          >
                            Hapus pilihan
                          </button>
                          <button
                            className="sample-action"
                            type="button"
                            onClick={() => {
                              setEditingFramesSampleId(null);
                              setSelectedFrameIndices(new Set());
                            }}
                            disabled={isTraining}
                          >
                            Batal
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
                  })}
                </section>
              ))}
            </div>
          )}
        </section>
      </div>
      {isDatabaseSettingsOpen && (
        <div className="database-settings-backdrop">
          <section className="database-settings" role="dialog" aria-modal="true" aria-labelledby="database-settings-title">
            <div className="database-settings-heading">
              <div>
                <h2 id="database-settings-title">Penyimpanan data</h2>
                <p>{user?.displayName ?? user?.email}</p>
              </div>
              <button className="sample-action" type="button" onClick={() => setIsDatabaseSettingsOpen(false)}>Tutup</button>
            </div>
            <div className="database-options" role="radiogroup" aria-label="Pilihan database">
              <label className={`database-option${settingsMode === 'shared' ? ' is-selected' : ''}`}>
                <input type="radio" name="database-mode" checked={settingsMode === 'shared'} onChange={() => setSettingsMode('shared')} />
                <span><strong>Database bersama</strong><small>Gunakan data latihan bersama pengguna lain.</small></span>
              </label>
              <label className={`database-option${settingsMode === 'personal' ? ' is-selected' : ''}`}>
                <input type="radio" name="database-mode" checked={settingsMode === 'personal'} onChange={() => setSettingsMode('personal')} />
                <span><strong>Database sendiri</strong><small>Simpan data di Firebase project milik Anda.</small></span>
              </label>
            </div>
            {settingsMode === 'personal' && (
              <div className="database-config-fields">
                <label>API key<input value={configDraft.apiKey} onChange={(event) => setConfigDraft({ ...configDraft, apiKey: event.target.value })} autoComplete="off" /></label>
                <label>Auth domain<input value={configDraft.authDomain} onChange={(event) => setConfigDraft({ ...configDraft, authDomain: event.target.value })} placeholder="project-id.firebaseapp.com" autoComplete="off" /></label>
                <label>App ID<input value={configDraft.appId} onChange={(event) => setConfigDraft({ ...configDraft, appId: event.target.value })} autoComplete="off" /></label>
                <label>Project ID<input value={configDraft.projectId} onChange={(event) => setConfigDraft({ ...configDraft, projectId: event.target.value })} autoComplete="off" /></label>
                <label>Realtime Database URL<input value={configDraft.databaseURL} onChange={(event) => setConfigDraft({ ...configDraft, databaseURL: event.target.value })} placeholder="https://project-id-default-rtdb..." autoComplete="off" /></label>
                <p className="database-config-note">Akun Google akan diminta untuk masuk ke Firebase project tersebut. Aktifkan Google Authentication dan izinkan akses baca/tulis `/samples` melalui Rules.</p>
              </div>
            )}
            {settingsMode === 'personal' && activeDatabase && personalFirebaseConfig && (
              <div className="database-catalog-section">
                <h3 className="database-catalog-title">Publikasikan ke Katalog</h3>
                <p className="database-config-note">Bagikan database ini ke katalog publik agar pengguna lain dapat menemukan dan menggunakannya.</p>
                <div className="catalog-publish-form">
                  <label>
                    <span>Nama Tampilan</span>
                    <input
                      type="text"
                      className="label-input"
                      value={catalogDraft.displayName}
                      onChange={(event) => setCatalogDraft({ ...catalogDraft, displayName: event.target.value })}
                      placeholder="Contoh: Database Gestur Bahasa Isyarat"
                      maxLength={64}
                      autoComplete="off"
                    />
                  </label>
                  <label>
                    <span>Deskripsi</span>
                    <textarea
                      className="label-input"
                      value={catalogDraft.description}
                      onChange={(event) => setCatalogDraft({ ...catalogDraft, description: event.target.value })}
                      placeholder="Jelaskan isi database ini (label apa saja, untuk siapa, dll.)"
                      maxLength={280}
                      rows={3}
                      style={{ resize: 'vertical' }}
                    />
                  </label>
                  <div className="catalog-publish-preview">
                    <strong>Pratinjau:</strong>
                    <div className="catalog-preview-stats">
                      <span>{usableSamples.length} sampel valid</span>
                      <span>{Object.keys(labelCounts).length} label</span>
                    </div>
                  </div>
                  {catalogPublishError && <p className="database-settings-error" role="alert">{catalogPublishError}</p>}
                  <div className="catalog-publish-actions">
                    <button
                      className="train-button"
                      type="button"
                      onClick={handlePublishCatalog}
                      disabled={isPublishingCatalog || !catalogDraft.displayName.trim() || usableSamples.length === 0}
                    >
                      {isPublishingCatalog ? 'Memublikasikan...' : 'Publikasikan ke Katalog'}
                    </button>
                  </div>
                </div>
              </div>
            )}
            {settingsError && <p className="database-settings-error" role="alert">{settingsError}</p>}
            <div className="database-settings-actions">
              <button className="sample-action" type="button" onClick={() => void signOut()}>Keluar</button>
              <button className="train-button" type="button" onClick={() => void saveDatabaseSettings()} disabled={isConnectingPersonalDatabase}>
                {isConnectingPersonalDatabase ? 'Menghubungkan...' : settingsMode === 'personal' ? 'Hubungkan & simpan' : 'Simpan pilihan'}
              </button>
            </div>
          </section>
        </div>
      )}
      {isImportDialogOpen && (
        <div className="database-settings-backdrop" onClick={closeImportDialog}>
          <section className="database-settings" role="dialog" aria-modal="true" aria-labelledby="import-dialog-title" onClick={(e) => e.stopPropagation()}>
            <div className="database-settings-heading">
              <div>
                <h2 id="import-dialog-title">Impor Data Gestur</h2>
                <p>Pilih file JSON hasil ekspor untuk menambahkan data latihan baru.</p>
              </div>
              <button className="sample-action" type="button" onClick={closeImportDialog}>Tutup</button>
            </div>
            <div className="database-config-fields">
              <label>
                <span>File JSON</span>
                <input
                  type="file"
                  accept=".json"
                  className="label-input"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void handleImport(file);
                  }}
                  disabled={isImporting}
                />
                <small>Format: file JSON dari fitur ekspor. Data dengan label yang sudah ada akan dilewati.</small>
              </label>
              {importError && <p className="database-settings-error" role="alert">{importError}</p>}
              {importResult && (
                <div className="catalog-publish-preview" style={{marginTop: 12}}>
                  <strong>Hasil Impor:</strong>
                  <div className="catalog-preview-stats">
                    <span>{importResult.successCount} ditambahkan</span>
                    <span>{importResult.skippedCount} dilewati</span>
                    <span>{importResult.errorCount} gagal</span>
                  </div>
                  {importResult.importedLabels.length > 0 && (
                    <div style={{marginTop: 8, fontSize: '0.875rem', color: 'var(--text-muted)'}}>
                      Label baru: {importResult.importedLabels.join(', ')}
                    </div>
                  )}
                  {importResult.errors.length > 0 && (
                    <details style={{marginTop: 8, fontSize: '0.875rem'}}>
                      <summary>Detail kesalahan ({importResult.errorCount})</summary>
                      <ul style={{marginTop: 4, paddingLeft: 20}}>
                        {importResult.errors.slice(0, 5).map((error, index) => (
                          <li key={`${index}-${error}`}>{error}</li>
                        ))}
                      </ul>
                      {importResult.errorCount > 5 && (
                        <small>{importResult.errorCount - 5} kesalahan lainnya tidak ditampilkan.</small>
                      )}
                    </details>
                  )}
                </div>
              )}
            </div>
            <div className="database-settings-actions">
              <button className="sample-action" type="button" onClick={closeImportDialog}>
                {importResult ? 'Selesai' : 'Batal'}
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
