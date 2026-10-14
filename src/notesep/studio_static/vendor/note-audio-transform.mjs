var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

// node_modules/@soundtouchjs/core/.dist/AbstractSamplePipe.js
var AbstractSamplePipe = class {
  /**
   * Constructs an AbstractSamplePipe.
   * @param options Constructor options.
   *
   * @remarks
   * When `createBuffers` is true, both factories are required so subclasses can
   * control exact buffer implementations without unsafe casting.
   */
  constructor({ createBuffers = false, inputBufferFactory, outputBufferFactory } = {}) {
    /**
     * Input buffer for audio samples.
     */
    __publicField(this, "_inputBuffer");
    /**
     * Output buffer for processed audio samples.
     */
    __publicField(this, "_outputBuffer");
    if (createBuffers) {
      if (!inputBufferFactory || !outputBufferFactory) {
        throw new Error("buffer factories are required when createBuffers is true");
      }
      this._inputBuffer = inputBufferFactory();
      this._outputBuffer = outputBufferFactory();
    } else {
      this._inputBuffer = null;
      this._outputBuffer = null;
    }
  }
  /**
   * Gets the input buffer.
   * @returns The current input buffer instance, or null if not set.
   */
  get inputBuffer() {
    return this._inputBuffer;
  }
  /**
   * Sets the input buffer.
   * @param inputBuffer The new input buffer instance, or null to unset.
   */
  set inputBuffer(inputBuffer) {
    this._inputBuffer = inputBuffer;
  }
  /**
   * Gets the output buffer.
   * @returns The current output buffer instance, or null if not set.
   */
  get outputBuffer() {
    return this._outputBuffer;
  }
  /**
   * Sets the output buffer.
   * @param outputBuffer The new output buffer instance, or null to unset.
   */
  set outputBuffer(outputBuffer) {
    this._outputBuffer = outputBuffer;
  }
  /**
   * Clears both input and output buffers.
   *
   * @remarks
   * Resets the state of both input and output buffers, if present, by calling their `clear()` methods.
   */
  clear() {
    this._inputBuffer?.clear();
    this._outputBuffer?.clear();
  }
};

// node_modules/@soundtouchjs/core/.dist/CircularSampleBuffer.js
var SAMPLES_PER_FRAME = 2;
var CircularSampleBuffer = class {
  /**
   * @param capacityFrames Initial frame capacity before automatic growth.
   */
  constructor(capacityFrames = 2048) {
    __publicField(this, "_buffer");
    __publicField(this, "_capacityFrames");
    __publicField(this, "_readFrame");
    __publicField(this, "_frameCount");
    const normalizedCapacity = Math.max(1, Math.floor(capacityFrames));
    this._capacityFrames = normalizedCapacity;
    this._buffer = new Float32Array(normalizedCapacity * SAMPLES_PER_FRAME);
    this._readFrame = 0;
    this._frameCount = 0;
  }
  /**
   * Allocated capacity expressed in frames.
   * @returns The number of frames the buffer can currently hold without resizing.
   */
  get capacityFrames() {
    return this._capacityFrames;
  }
  /**
   * Number of buffered frames currently readable.
   * @returns The number of frames available for reading.
   */
  get frameCount() {
    return this._frameCount;
  }
  /**
   * Clears the buffer without shrinking allocated capacity.
   *
   * @remarks
   * Resets the read cursor and frame count, but does not deallocate the underlying storage.
   */
  clear() {
    this._readFrame = 0;
    this._frameCount = 0;
  }
  /**
   * Ensures the internal storage can hold at least `minCapacityFrames`.
   *
   * @param minCapacityFrames Minimum frame capacity required.
   * @remarks
   * Grows the buffer if needed, preserving all readable frames in order.
   */
  ensureCapacity(minCapacityFrames) {
    const normalizedMinCapacityFrames = Math.max(0, Math.floor(minCapacityFrames));
    if (normalizedMinCapacityFrames <= this._capacityFrames) {
      return;
    }
    const nextCapacity = Math.max(normalizedMinCapacityFrames, this._capacityFrames * 2, this._capacityFrames + 1024);
    const nextBuffer = new Float32Array(nextCapacity * SAMPLES_PER_FRAME);
    for (let frame = 0; frame < this._frameCount; frame += 1) {
      const sourceFrame = (this._readFrame + frame) % this._capacityFrames;
      const sourceIndex = sourceFrame * SAMPLES_PER_FRAME;
      const destIndex = frame * SAMPLES_PER_FRAME;
      nextBuffer[destIndex] = this._buffer[sourceIndex];
      nextBuffer[destIndex + 1] = this._buffer[sourceIndex + 1];
    }
    this._buffer = nextBuffer;
    this._capacityFrames = nextCapacity;
    this._readFrame = 0;
  }
  /**
   * Appends source frames to the end of the ring.
   *
   * @param source Interleaved stereo source samples.
   * @param sourceFrameOffset Source offset in frames.
   * @param frameCount Number of frames to append; defaults to all complete remaining frames.
   * @remarks
   * Automatically grows the buffer if needed. Only complete frames are appended.
   */
  pushSamples(source, sourceFrameOffset = 0, frameCount = 0) {
    const normalizedSourceFrameOffset = Math.max(0, Math.floor(sourceFrameOffset));
    const sourceStartSample = normalizedSourceFrameOffset * SAMPLES_PER_FRAME;
    const availableFrames = Math.max(0, Math.floor((source.length - sourceStartSample) / SAMPLES_PER_FRAME));
    const requestedFrames = frameCount > 0 ? Math.floor(frameCount) : 0;
    const framesToWrite = requestedFrames > 0 ? Math.min(requestedFrames, availableFrames) : availableFrames;
    if (framesToWrite <= 0) {
      return;
    }
    this.ensureCapacity(this._frameCount + framesToWrite);
    const writeFrame = (this._readFrame + this._frameCount) % this._capacityFrames;
    for (let frame = 0; frame < framesToWrite; frame += 1) {
      const sourceIndex = sourceStartSample + frame * SAMPLES_PER_FRAME;
      const destFrame = (writeFrame + frame) % this._capacityFrames;
      const destIndex = destFrame * SAMPLES_PER_FRAME;
      this._buffer[destIndex] = source[sourceIndex];
      this._buffer[destIndex + 1] = source[sourceIndex + 1];
    }
    this._frameCount += framesToWrite;
  }
  /**
   * Contract alias for `pushSamples`.
   *
   * @param source Interleaved stereo source samples.
   * @param sourceFrameOffset Source offset in frames.
   * @param frameCount Number of frames to append.
   */
  putSamples(source, sourceFrameOffset = 0, frameCount = 0) {
    this.pushSamples(source, sourceFrameOffset, frameCount);
  }
  /**
   * Extracts frames from the ring into `target`.
   *
   * @param target Destination array for interleaved stereo samples.
   * @param sourceFrameOffset Read offset in frames.
   * @param frameCount Number of frames requested.
   * @param consume When true, consumed frames are dropped from the front.
   * @returns Number of frames copied.
   * @remarks
   * If `consume` is true, the extracted frames are removed from the buffer.
   */
  extract(target, sourceFrameOffset = 0, frameCount = 0, consume = false) {
    const normalizedSourceFrameOffset = Math.max(0, Math.floor(sourceFrameOffset));
    const requestedFrames = frameCount > 0 ? Math.floor(frameCount) : 0;
    const framesAvailable = Math.max(0, this._frameCount - normalizedSourceFrameOffset);
    const framesToRead = requestedFrames > 0 ? Math.min(requestedFrames, framesAvailable) : framesAvailable;
    if (framesToRead <= 0) {
      return 0;
    }
    for (let frame = 0; frame < framesToRead; frame += 1) {
      const sourceFrame = (this._readFrame + normalizedSourceFrameOffset + frame) % this._capacityFrames;
      const sourceIndex = sourceFrame * SAMPLES_PER_FRAME;
      const targetIndex = frame * SAMPLES_PER_FRAME;
      target[targetIndex] = this._buffer[sourceIndex];
      target[targetIndex + 1] = this._buffer[sourceIndex + 1];
    }
    if (consume) {
      const framesToDrop = normalizedSourceFrameOffset + framesToRead;
      this.dropFrames(framesToDrop);
    }
    return framesToRead;
  }
  /**
   * Reads a single sample value by logical sample index.
   *
   * @param sampleIndex Logical sample index relative to the readable head.
   * @returns Sample value, or `0` when the index falls outside readable data.
   * @remarks
   * Used for random access to individual samples within the readable region.
   */
  readSample(sampleIndex) {
    const normalizedSampleIndex = Math.max(0, Math.floor(sampleIndex));
    const frameOffset = Math.floor(normalizedSampleIndex / SAMPLES_PER_FRAME);
    if (frameOffset >= this._frameCount) {
      return 0;
    }
    const channelOffset = normalizedSampleIndex % SAMPLES_PER_FRAME;
    const sourceFrame = (this._readFrame + frameOffset) % this._capacityFrames;
    const sourceIndex = sourceFrame * SAMPLES_PER_FRAME + channelOffset;
    return this._buffer[sourceIndex] ?? 0;
  }
  /**
   * Drops frames from the front of the ring.
   *
   * @param frameCount Maximum number of frames to remove.
   * @returns Number of frames removed.
   * @remarks
   * Advances the read cursor and reduces the frame count. If all frames are dropped, resets the read cursor.
   */
  dropFrames(frameCount) {
    const normalizedFrameCount = Math.max(0, Math.floor(frameCount));
    const framesToDrop = Math.max(0, Math.min(normalizedFrameCount, this._frameCount));
    if (framesToDrop === 0) {
      return 0;
    }
    this._readFrame = (this._readFrame + framesToDrop) % this._capacityFrames;
    this._frameCount -= framesToDrop;
    if (this._frameCount === 0) {
      this._readFrame = 0;
    }
    return framesToDrop;
  }
  /**
   * Contract alias for `dropFrames`.
   *
   * @param frameCount Number of frames to consume.
   */
  receive(frameCount = this._frameCount) {
    this.dropFrames(frameCount);
  }
};

// node_modules/@soundtouchjs/core/.dist/FifoSampleBuffer.js
var BYTES_PER_SAMPLE = 4;
var SAMPLES_PER_FRAME2 = 2;
var BYTES_PER_FRAME = BYTES_PER_SAMPLE * SAMPLES_PER_FRAME2;
var DEFAULT_MAX_FRAMES = 131072;
var FifoSampleBuffer = class {
  /**
   * Creates a new FifoSampleBuffer.
   * @param maxFrames Maximum number of frames for buffer allocation.
   */
  constructor(maxFrames = DEFAULT_MAX_FRAMES) {
    /**
     * Backing ArrayBuffer for sample storage.
     * @remarks
     * Underlying memory for the buffer, which may be resized as needed.
     */
    __publicField(this, "_buffer");
    /**
     * Float32Array view of the buffer.
     * @remarks
     * Provides direct access to the sample data for reading and writing.
     */
    __publicField(this, "_vector");
    /**
     * Current read position (frame index).
     * @remarks
     * Indicates the logical start of readable data within the buffer.
     */
    __publicField(this, "_position");
    /**
     * Number of frames currently stored.
     * @remarks
     * Represents the number of complete stereo frames available for reading.
     */
    __publicField(this, "_frameCount");
    this._buffer = new ArrayBuffer(0, {
      maxByteLength: maxFrames * BYTES_PER_FRAME
    });
    this._vector = new Float32Array(this._buffer);
    this._position = 0;
    this._frameCount = 0;
  }
  /**
   * Returns the Float32Array view of the buffer.
   * @returns The Float32Array containing the sample data.
   */
  get vector() {
    return this._vector;
  }
  /**
   * Returns the current read position (frame index).
   * @returns The current frame index for reading.
   */
  get position() {
    return this._position;
  }
  /**
   * Returns the start sample index for reading.
   * @returns The sample index corresponding to the start of readable data.
   */
  get startIndex() {
    return this._position * 2;
  }
  /**
   * Returns the number of frames currently stored.
   * @returns The number of complete frames available for reading.
   */
  get frameCount() {
    return this._frameCount;
  }
  /**
   * Returns the end sample index for reading.
   * @returns The sample index corresponding to the end of readable data.
   */
  get endIndex() {
    return (this._position + this._frameCount) * 2;
  }
  /**
   * Clears the buffer and resets position and frame count.
   * @remarks
   * Fills the buffer with zeros and resets all internal state.
   */
  clear() {
    this._vector.fill(0);
    this._position = 0;
    this._frameCount = 0;
  }
  /**
   * Adds empty frames to the buffer.
   * @param numFrames Number of frames to add.
   */
  put(numFrames) {
    this._frameCount += numFrames;
  }
  /**
   * Adds samples to the buffer from a Float32Array.
   * @param samples Source samples (interleaved stereo).
   * @param position Start frame index in source.
   * @param numFrames Number of frames to copy (default: all available).
   * @remarks
   * Automatically grows the buffer if needed. Only complete frames are appended.
   */
  putSamples(samples, position = 0, numFrames = 0) {
    const sourceOffset = position * 2;
    if (!(numFrames >= 0) || numFrames === 0) {
      numFrames = (samples.length - sourceOffset) / 2;
    }
    const numSamples = numFrames * 2;
    this.ensureCapacity(numFrames + this._frameCount);
    const destOffset = this.endIndex;
    this._vector.set(samples.subarray(sourceOffset, sourceOffset + numSamples), destOffset);
    this._frameCount += numFrames;
  }
  /**
   * Adds samples from another FifoSampleBuffer.
   * @param buffer Source buffer.
   * @param position Start frame index in source buffer.
   * @param numFrames Number of frames to copy (default: all available).
   */
  putBuffer(buffer, position = 0, numFrames = 0) {
    if (!(numFrames >= 0) || numFrames === 0) {
      numFrames = buffer.frameCount - position;
    }
    this.putSamples(buffer.vector, buffer.position + position, numFrames);
  }
  /**
   * Advances the read position and reduces frame count.
   * @param numFrames Number of frames to receive (default: all available).
   * @remarks
   * Consumed frames are no longer available for reading.
   */
  receive(numFrames) {
    if (numFrames === void 0 || !(numFrames >= 0) || numFrames > this._frameCount) {
      numFrames = this._frameCount;
    }
    this._frameCount -= numFrames;
    this._position += numFrames;
  }
  /**
   * Copies and receives samples into an output array.
   * @param output Destination Float32Array.
   * @param numFrames Number of frames to copy and receive.
   * @remarks
   * Advances the read position after copying.
   */
  receiveSamples(output, numFrames = 0) {
    const numSamples = numFrames * 2;
    const sourceOffset = this.startIndex;
    output.set(this._vector.subarray(sourceOffset, sourceOffset + numSamples));
    this.receive(numFrames);
  }
  /**
   * Extracts samples into an output array without advancing position.
   * @param output Destination Float32Array.
   * @param position Start frame index in buffer.
   * @param numFrames Number of frames to extract.
   */
  extract(output, position = 0, numFrames = 0) {
    const sourceOffset = this.startIndex + position * 2;
    const numSamples = numFrames * 2;
    output.set(this._vector.subarray(sourceOffset, sourceOffset + numSamples));
  }
  /**
   * Ensures the buffer has capacity for at least numFrames.
   * @param numFrames Minimum number of frames required.
   * @remarks
   * Grows the buffer if needed, preserving all readable frames in order.
   */
  ensureCapacity(numFrames = 0) {
    const minLength = Math.floor(numFrames * SAMPLES_PER_FRAME2);
    if (this._vector.length < minLength) {
      const newByteLength = minLength * BYTES_PER_SAMPLE;
      if (newByteLength <= this._buffer.maxByteLength) {
        this.rewind();
        this._buffer.resize(newByteLength);
        this._vector = new Float32Array(this._buffer);
      } else {
        const newMaxBytes = newByteLength * 2;
        const newBuffer = new ArrayBuffer(newByteLength, {
          maxByteLength: newMaxBytes
        });
        const newVector = new Float32Array(newBuffer);
        newVector.set(this._vector.subarray(this.startIndex, this.endIndex));
        this._buffer = newBuffer;
        this._vector = newVector;
        this._position = 0;
      }
    } else {
      this.rewind();
    }
  }
  /**
   * Ensures buffer has capacity for additional frames.
   * @param numFrames Number of additional frames required.
   */
  ensureAdditionalCapacity(numFrames = 0) {
    this.ensureCapacity(this._frameCount + numFrames);
  }
  /**
   * Moves all unread samples to the start of the buffer.
   * @remarks
   * Compacts the buffer so that all unread samples are at the beginning, freeing space for new data.
   */
  rewind() {
    if (this._position > 0) {
      this._vector.set(this._vector.subarray(this.startIndex, this.endIndex));
      this._position = 0;
    }
  }
};

// node_modules/@soundtouchjs/core/.dist/SampleBufferAdapter.js
var FifoSampleBufferAdapter = class {
  constructor() {
    __publicField(this, "inputBuffer");
    this.inputBuffer = null;
  }
  get frameCount() {
    return this.inputBuffer?.frameCount ?? 0;
  }
  clear() {
    this.inputBuffer = null;
  }
  syncFromInputBuffer(inputBuffer) {
    this.inputBuffer = inputBuffer;
  }
  extract(target, sourceFrameOffset, frameCount) {
    const buffer = this.inputBuffer;
    if (buffer === null) {
      return 0;
    }
    const availableFrames = Math.max(0, buffer.frameCount - sourceFrameOffset);
    const framesToExtract = Math.max(0, Math.min(frameCount, availableFrames));
    if (framesToExtract === 0) {
      return 0;
    }
    buffer.extract(target, sourceFrameOffset, framesToExtract);
    return framesToExtract;
  }
  receive(frameCount) {
    this.inputBuffer?.receive(frameCount);
  }
};
var CircularSampleBufferAdapter = class {
  constructor() {
    __publicField(this, "circularBuffer");
    __publicField(this, "scratch");
    this.circularBuffer = new CircularSampleBuffer();
    this.scratch = new Float32Array(0);
  }
  get frameCount() {
    return this.circularBuffer.frameCount;
  }
  clear() {
    this.circularBuffer.clear();
  }
  syncFromInputBuffer(inputBuffer) {
    if (inputBuffer instanceof FifoSampleBuffer) {
      const frames2 = inputBuffer.frameCount;
      if (frames2 === 0) {
        return;
      }
      this.circularBuffer.pushSamples(inputBuffer.vector, inputBuffer.position, frames2);
      inputBuffer.receive(frames2);
      return;
    }
    const frames = inputBuffer.frameCount;
    if (frames === 0) {
      return;
    }
    const sampleCount = frames * 2;
    if (this.scratch.length < sampleCount) {
      this.scratch = new Float32Array(sampleCount);
    }
    inputBuffer.extract(this.scratch, 0, frames);
    this.circularBuffer.pushSamples(this.scratch, 0, frames);
    inputBuffer.receive(frames);
  }
  extract(target, sourceFrameOffset, frameCount) {
    return this.circularBuffer.extract(target, sourceFrameOffset, frameCount, false);
  }
  receive(frameCount) {
    this.circularBuffer.dropFrames(frameCount);
  }
};
var createFifoSampleBufferAdapter = () => new FifoSampleBufferAdapter();
var createCircularSampleBufferAdapter = () => new CircularSampleBufferAdapter();

// node_modules/@soundtouchjs/interpolation-strategy-lanczos/.dist/index.js
var LANCZOS_DEFAULT_PARAMS = {
  zeroCrossings: 4,
  normalize: false
};
function normalizeLanczosParams(params, defaults) {
  const merged = {
    ...defaults,
    ...params ?? {}
  };
  const zeroCrossings = Math.max(2, Math.min(8, Math.round(Number(merged["zeroCrossings"] ?? defaults["zeroCrossings"] ?? 4))));
  const normalize = Boolean(merged["normalize"]);
  return { zeroCrossings, normalize };
}
function applyLanczosParams(state, params) {
  if (typeof state !== "object" || state === null) {
    return;
  }
  const record = state;
  record.params = {
    zeroCrossings: Math.max(2, Math.round(Number(params["zeroCrossings"] ?? 4))),
    normalize: Boolean(params["normalize"])
  };
}
function readFrameSample(src, srcOffset, numFrames, frameIndex, channel, state) {
  if (frameIndex < 0) {
    return channel === 0 ? state.prevSampleL : state.prevSampleR;
  }
  if (frameIndex >= numFrames) {
    const edgeIndex = srcOffset + 2 * (numFrames - 1) + channel;
    return src[edgeIndex];
  }
  return src[srcOffset + 2 * frameIndex + channel];
}
function normalizedSinc(x) {
  if (x === 0) {
    return 1;
  }
  const value = Math.PI * x;
  return Math.sin(value) / value;
}
function lanczosWeight(distance, radius) {
  const absDistance = Math.abs(distance);
  if (absDistance >= radius) {
    return 0;
  }
  return normalizedSinc(distance) * normalizedSinc(distance / radius);
}
var lanczosKernel = (src, srcOffset, numFrames, position, channel, state) => {
  const kernelState = state;
  const radius = kernelState.params.zeroCrossings;
  const normalize = Boolean(kernelState.params.normalize);
  const center = Math.floor(position);
  const start = center - (radius - 1);
  const end = center + radius;
  let numerator = 0;
  let denominator = 0;
  for (let sampleIndex = start; sampleIndex <= end; sampleIndex += 1) {
    const distance = position - sampleIndex;
    const weight = lanczosWeight(distance, radius);
    numerator += readFrameSample(src, srcOffset, numFrames, sampleIndex, channel, kernelState) * weight;
    denominator += weight;
  }
  if (Math.abs(denominator) < 1e-12) {
    return readFrameSample(src, srcOffset, numFrames, Math.round(position), channel, kernelState);
  }
  return normalize ? numerator / denominator : numerator / (denominator || 1);
};
lanczosKernel.createState = () => ({
  prevSampleL: 0,
  prevSampleR: 0,
  params: { ...LANCZOS_DEFAULT_PARAMS }
});
var lanczosStrategy = {
  id: "lanczos",
  baseStrategy: "linear",
  kernel: lanczosKernel,
  defaultParams: LANCZOS_DEFAULT_PARAMS,
  normalizeParams: normalizeLanczosParams,
  applyParams: applyLanczosParams
};

// node_modules/@soundtouchjs/core/.dist/interpolationStrategyRegistry.js
var strategyRegistry = /* @__PURE__ */ new Map();
var activeStrategyId = "lanczos";
function readStrategySelection(strategy) {
  if (typeof strategy === "string") {
    return { id: strategy };
  }
  if (strategy !== void 0) {
    return strategy;
  }
  return { id: activeStrategyId };
}
function requireRegisteredStrategy(strategyId) {
  const registered = strategyRegistry.get(strategyId);
  if (registered !== void 0) {
    return registered;
  }
  throw new Error(`Unknown interpolation strategy id "${strategyId}". Register it before use.`);
}
function registerBuiltInInterpolationStrategy(registration) {
  const baseStrategy = registration.baseStrategy ?? "lanczos";
  strategyRegistry.set(registration.id, {
    id: registration.id,
    baseStrategy,
    builtIn: true,
    kernel: registration.kernel,
    defaultParams: { ...registration.defaultParams ?? {} },
    normalizeParams: registration.normalizeParams,
    applyParams: registration.applyParams
  });
}
function resolveKernelRegistration(registration, visited = /* @__PURE__ */ new Set()) {
  if (registration.kernel !== void 0) {
    return registration;
  }
  if (visited.has(registration.id)) {
    throw new Error(`Interpolation strategy resolution cycle detected at "${registration.id}".`);
  }
  visited.add(registration.id);
  const next = requireRegisteredStrategy(registration.baseStrategy);
  return resolveKernelRegistration(next, visited);
}
function normalizeParams(registration, params) {
  const defaults = registration.defaultParams;
  if (registration.normalizeParams !== void 0) {
    return registration.normalizeParams(params, defaults);
  }
  const normalized = { ...defaults };
  if (params !== void 0) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== void 0) {
        normalized[key] = value;
      }
    }
  }
  return normalized;
}
function resolveInterpolationStrategyRuntime(strategy) {
  const selection = readStrategySelection(strategy);
  const registered = requireRegisteredStrategy(selection.id);
  const kernelRegistration = resolveKernelRegistration(registered);
  const kernel = kernelRegistration.kernel;
  if (kernel === void 0) {
    throw new Error(`Interpolation strategy "${selection.id}" did not resolve to a kernel.`);
  }
  const params = normalizeParams(registered, selection.params);
  return {
    id: registered.id,
    kernel,
    params,
    applyParams: registered.applyParams ?? kernelRegistration.applyParams
  };
}
registerBuiltInInterpolationStrategy({
  ...lanczosStrategy,
  baseStrategy: "lanczos"
});

// node_modules/@soundtouchjs/core/.dist/RateTransposer.js
var RateTransposer = class _RateTransposer extends AbstractSamplePipe {
  /**
   * Creates a RateTransposer instance.
   * @param options Constructor options.
   * @remarks
   * Accepts factories for buffer and adapter creation, and allows specifying the interpolation strategy.
   */
  constructor({ createBuffers = false, sampleBufferAdapterFactory = createCircularSampleBufferAdapter, sampleBufferFactory = () => new CircularSampleBuffer(), interpolationStrategy } = {}) {
    super({
      createBuffers,
      inputBufferFactory: sampleBufferFactory,
      outputBufferFactory: sampleBufferFactory
    });
    /**
     * Current rate factor for transposition.
     */
    __publicField(this, "_rate");
    /**
     * Source position (in frames) for the next output sample, relative to the
     * current processing block where 0 is the first frame and -1 is prevSample.
     */
    __publicField(this, "fractionalPosition");
    /**
     * Previous left channel sample for interpolation.
     */
    __publicField(this, "previousLeftSample");
    /**
     * Previous right channel sample for interpolation.
     */
    __publicField(this, "previousRightSample");
    /** Scratch space used for extracted input samples. */
    __publicField(this, "inputScratch");
    /** Scratch space used for generated output samples. */
    __publicField(this, "outputScratch");
    /** Factory used when cloning or initializing adapter strategy. */
    __publicField(this, "sampleBufferAdapterFactory");
    /** Factory used to construct input/output chain buffers. */
    __publicField(this, "sampleBufferFactory");
    /** Adapter that normalizes reads from the bound input buffer. */
    __publicField(this, "inputAdapter");
    /** Selected interpolation strategy for transposition. */
    __publicField(this, "interpolationStrategy");
    /** Resolved kernel used by this transposer instance. */
    __publicField(this, "resolvedInterpolationKernel");
    /** Optional per-instance state for plugin kernels. */
    __publicField(this, "kernelState");
    /** Normalized params for the selected interpolation strategy. */
    __publicField(this, "interpolationStrategyParams");
    /** Optional params application hook from the strategy registration. */
    __publicField(this, "applyKernelParams");
    this.fractionalPosition = -1;
    this.previousLeftSample = 0;
    this.previousRightSample = 0;
    this._rate = 1;
    this.inputScratch = new Float32Array(0);
    this.outputScratch = new Float32Array(0);
    this.sampleBufferAdapterFactory = sampleBufferAdapterFactory;
    this.sampleBufferFactory = sampleBufferFactory;
    this.inputAdapter = sampleBufferAdapterFactory();
    this.interpolationStrategy = "lanczos";
    this.resolvedInterpolationKernel = () => 0;
    this.kernelState = void 0;
    this.interpolationStrategyParams = {};
    this.applyKernelParams = void 0;
    this.setInterpolationStrategy(interpolationStrategy ?? "lanczos");
  }
  /**
   * Sets the rate factor for transposition.
   * @param rate Rate factor.
   */
  set rate(rate) {
    this._rate = rate;
  }
  /**
   * Active interpolation strategy.
   * @returns The current interpolation strategy identifier.
   */
  get strategy() {
    return this.interpolationStrategy;
  }
  /**
   * Active interpolation strategy params.
   * @returns The current interpolation strategy parameters.
   */
  get strategyParams() {
    return { ...this.interpolationStrategyParams };
  }
  /**
   * Switches interpolation strategy at runtime.
   * @param strategy The new interpolation strategy to use.
   */
  setInterpolationStrategy(strategy) {
    const resolved = resolveInterpolationStrategyRuntime(strategy);
    this.interpolationStrategy = resolved.id;
    this.resolvedInterpolationKernel = resolved.kernel;
    this.interpolationStrategyParams = { ...resolved.params };
    this.applyKernelParams = resolved.applyParams;
    if ("createState" in this.resolvedInterpolationKernel && typeof this.resolvedInterpolationKernel.createState === "function") {
      this.kernelState = this.resolvedInterpolationKernel.createState();
    } else {
      this.kernelState = void 0;
    }
    if (this.applyKernelParams !== void 0) {
      this.applyKernelParams(this.kernelState, this.interpolationStrategyParams);
    }
    this.reset();
  }
  /**
   * Applies a partial params update to the current interpolation strategy.
   * @param params Partial set of parameters to update.
   */
  setInterpolationStrategyParams(params) {
    const nextParams = {
      ...this.interpolationStrategyParams
    };
    for (const [key, value] of Object.entries(params)) {
      if (value !== void 0) {
        nextParams[key] = value;
      }
    }
    this.interpolationStrategyParams = nextParams;
    if (this.applyKernelParams !== void 0) {
      this.applyKernelParams(this.kernelState, this.interpolationStrategyParams);
    }
  }
  /**
   * Resets internal state for interpolation.
   * @remarks
   * Clears previous sample values and resets the fractional position for output generation.
   */
  reset() {
    this.fractionalPosition = -1;
    this.previousLeftSample = 0;
    this.previousRightSample = 0;
  }
  /**
   * Clears buffers and resets internal state.
   * @remarks
   * Calls clear on all internal buffers and resets interpolation state.
   */
  clear() {
    super.clear();
    this.inputAdapter.clear();
    this.reset();
  }
  /**
   * Creates a clone of this RateTransposer with the same rate.
   * @returns Cloned RateTransposer instance.
   */
  clone() {
    const result = new _RateTransposer({
      createBuffers: false,
      sampleBufferAdapterFactory: this.sampleBufferAdapterFactory,
      sampleBufferFactory: this.sampleBufferFactory,
      interpolationStrategy: {
        id: this.interpolationStrategy,
        params: this.interpolationStrategyParams
      }
    });
    result.rate = this._rate;
    return result;
  }
  /**
   * Processes input buffer and writes transposed samples to output buffer.
   * @remarks
   * Reads frames from the input buffer, applies rate transposition, and writes to the output buffer.
   */
  process() {
    if (this._inputBuffer === null || this._outputBuffer === null) {
      return;
    }
    this.inputAdapter.syncFromInputBuffer(this._inputBuffer);
    const numFrames = this.inputAdapter.frameCount;
    if (numFrames === 0) {
      return;
    }
    const numFramesOutput = this.transpose(numFrames);
    this.inputAdapter.receive(numFrames);
    if (numFramesOutput > 0) {
      this._outputBuffer.putSamples(this.outputScratch, 0, numFramesOutput);
    }
  }
  /**
   * Ensures temporary scratch arrays are large enough for the current frame request and estimated output size.
   *
   * @param numInputFrames Number of input frames that will be processed.
   * @remarks
   * Allocates or resizes scratch arrays as needed for efficient processing.
   */
  ensureScratchCapacity(numInputFrames) {
    const inputSamples = numInputFrames * 2;
    if (this.inputScratch.length < inputSamples) {
      this.inputScratch = new Float32Array(inputSamples);
    }
    const estimatedOutputFrames = Math.ceil(numInputFrames / this._rate) + 2;
    const outputSamples = Math.max(0, estimatedOutputFrames) * 2;
    if (this.outputScratch.length < outputSamples) {
      this.outputScratch = new Float32Array(outputSamples);
    }
  }
  /**
   * Transposes input samples by the current rate.
   * @param numFrames Number of input frames to transpose.
   * @returns Number of output frames written.
   * @remarks
   * Applies the selected interpolation kernel to generate output samples at the new rate.
   */
  transpose(numFrames = 0) {
    if (this._inputBuffer !== null) {
      this.inputAdapter.syncFromInputBuffer(this._inputBuffer);
      if (numFrames === 0) {
        numFrames = this.inputAdapter.frameCount;
      }
    }
    if (numFrames === 0) {
      return 0;
    }
    this.ensureScratchCapacity(numFrames);
    const src = this.inputScratch;
    const extractedFrames = this.inputAdapter.extract(src, 0, numFrames);
    if (extractedFrames === 0) {
      return 0;
    }
    numFrames = extractedFrames;
    return this.transposePluginKernel(numFrames);
  }
  /**
   * Handles transposition using a plugin kernel.
   * @remarks
   * Invokes the selected interpolation kernel for each output sample.
   */
  transposePluginKernel(numFrames) {
    const src = this.inputScratch;
    const dest = this.outputScratch;
    const srcOffset = 0;
    const destOffset = 0;
    const kernel = this.resolvedInterpolationKernel;
    const state = this.kernelState;
    const stateRecord = this.getKernelStateRecord(state);
    if (stateRecord !== void 0) {
      stateRecord.prevSampleL = this.previousLeftSample;
      stateRecord.prevSampleR = this.previousRightSample;
    }
    let i = 0;
    let position = this.fractionalPosition;
    const maxPosition = numFrames - 1;
    while (position <= maxPosition) {
      dest[destOffset + 2 * i] = kernel(src, srcOffset, numFrames, position, 0, state);
      dest[destOffset + 2 * i + 1] = kernel(src, srcOffset, numFrames, position, 1, state);
      i = i + 1;
      position += this._rate;
    }
    this.fractionalPosition = position - numFrames;
    this.previousLeftSample = src[srcOffset + 2 * numFrames - 2];
    this.previousRightSample = src[srcOffset + 2 * numFrames - 1];
    if (stateRecord !== void 0) {
      stateRecord.prevSampleL = this.previousLeftSample;
      stateRecord.prevSampleR = this.previousRightSample;
    }
    return i;
  }
  /**
   * Returns the kernel state record if available.
   * @param state The kernel state object.
   * @returns The state record with previous sample values, or undefined if not present.
   */
  getKernelStateRecord(state) {
    if (typeof state !== "object" || state === null) {
      return void 0;
    }
    const record = state;
    const prevSampleL = record["prevSampleL"];
    const prevSampleR = record["prevSampleR"];
    if (typeof prevSampleL === "number" && typeof prevSampleR === "number") {
      return record;
    }
    return void 0;
  }
};

// node_modules/@soundtouchjs/core/.dist/Stretch.js
var FifoStretchBufferAdapter = class {
  constructor() {
    __publicField(this, "buffer");
    __publicField(this, "fallbackBuffer");
    __publicField(this, "fallbackScratch");
    this.buffer = null;
    this.fallbackBuffer = new FifoSampleBuffer();
    this.fallbackScratch = new Float32Array(0);
  }
  /**
   * @param buffer Source buffer to expose through FIFO-style reads.
   */
  setBuffer(buffer) {
    if (buffer instanceof FifoSampleBuffer) {
      this.buffer = buffer;
      return;
    }
    const frameCount = buffer.frameCount;
    if (frameCount > 0) {
      const sampleCount = frameCount * 2;
      if (this.fallbackScratch.length < sampleCount) {
        this.fallbackScratch = new Float32Array(sampleCount);
      }
      buffer.extract(this.fallbackScratch, 0, frameCount);
      this.fallbackBuffer.clear();
      this.fallbackBuffer.putSamples(this.fallbackScratch, 0, frameCount);
      buffer.receive(frameCount);
    } else {
      this.fallbackBuffer.clear();
    }
    this.buffer = this.fallbackBuffer;
  }
  /**
   * Returns the currently bound FIFO buffer.
   * @throws Error when `setBuffer` has not been called yet.
   */
  getBoundBuffer() {
    if (this.buffer === null) {
      throw new Error("buffer is not set");
    }
    return this.buffer;
  }
  get frameCount() {
    return this.getBoundBuffer().frameCount;
  }
  get startIndex() {
    return this.getBoundBuffer().startIndex;
  }
  readSample(sampleIndex) {
    const boundBuffer = this.getBoundBuffer();
    const start = boundBuffer.startIndex;
    const end = start + boundBuffer.frameCount * 2;
    if (sampleIndex < start || sampleIndex >= end) {
      return 0;
    }
    return boundBuffer.vector[sampleIndex];
  }
  readSubarray(start, end) {
    return this.getBoundBuffer().vector.subarray(start, end);
  }
  receive(numFrames) {
    this.getBoundBuffer().receive(numFrames);
  }
  receiveSamples(output, numFrames) {
    this.getBoundBuffer().receiveSamples(output, numFrames);
  }
};
var GenericStretchWriteBufferAdapter = class {
  constructor() {
    __publicField(this, "buffer");
    this.buffer = null;
  }
  setOutputBuffer(buffer) {
    this.buffer = buffer;
  }
  /**
   * Returns the currently bound output buffer.
   * @throws Error when `setOutputBuffer` has not been called.
   */
  getBoundBuffer() {
    if (this.buffer === null) {
      throw new Error("output buffer is not set");
    }
    return this.buffer;
  }
  appendSamples(samples, numFrames) {
    this.getBoundBuffer().putSamples(samples, 0, numFrames);
  }
  putFrom(source, position, numFrames) {
    const sourceStart = source.startIndex + position * 2;
    const sourceEnd = sourceStart + numFrames * 2;
    const chunk = source.readSubarray(sourceStart, sourceEnd);
    this.getBoundBuffer().putSamples(chunk, 0, numFrames);
  }
};
var CircularStretchInputBufferAdapter = class {
  constructor() {
    __publicField(this, "circularBuffer");
    __publicField(this, "rangeScratch");
    this.circularBuffer = new CircularSampleBuffer();
    this.rangeScratch = new Float32Array(0);
  }
  /**
   * Binds a source buffer and stages its readable frames into the internal
   * circular storage.
   *
   * @param buffer Source buffer to import.
   */
  setBuffer(buffer) {
    if (buffer instanceof FifoSampleBuffer) {
      const frames2 = buffer.frameCount;
      if (frames2 > 0) {
        this.circularBuffer.pushSamples(buffer.vector, buffer.position, frames2);
        buffer.receive(frames2);
      }
      return;
    }
    const frames = buffer.frameCount;
    if (frames > 0) {
      const sampleCount = frames * 2;
      if (this.rangeScratch.length < sampleCount) {
        this.rangeScratch = new Float32Array(sampleCount);
      }
      buffer.extract(this.rangeScratch, 0, frames);
      this.circularBuffer.pushSamples(this.rangeScratch, 0, frames);
      buffer.receive(frames);
    }
  }
  get frameCount() {
    return this.circularBuffer.frameCount;
  }
  get startIndex() {
    return 0;
  }
  readSample(sampleIndex) {
    return this.circularBuffer.readSample(sampleIndex);
  }
  /**
   * Returns a contiguous range from circular storage, padding trailing values
   * with zeros when the requested range extends past available data.
   */
  readSubarray(start, end) {
    const normalizedStart = Math.max(0, Math.floor(start));
    const normalizedEnd = Math.max(normalizedStart, Math.floor(end));
    const requestedSamples = normalizedEnd - normalizedStart;
    const requestedFrames = Math.floor(requestedSamples / 2);
    if (requestedFrames <= 0) {
      return this.rangeScratch.subarray(0, 0);
    }
    const needed = requestedFrames * 2;
    if (this.rangeScratch.length < needed) {
      this.rangeScratch = new Float32Array(needed);
    }
    const sourceFrameOffset = Math.floor(normalizedStart / 2);
    const readFrames = this.circularBuffer.extract(this.rangeScratch, sourceFrameOffset, requestedFrames, false);
    const readSamples = readFrames * 2;
    if (readSamples < needed) {
      this.rangeScratch.fill(0, readSamples, needed);
    }
    return this.rangeScratch.subarray(0, needed);
  }
  receive(numFrames) {
    this.circularBuffer.dropFrames(numFrames);
  }
  receiveSamples(output, numFrames) {
    this.circularBuffer.extract(output, 0, numFrames, true);
  }
};
var createFifoStretchInputBufferAdapter = () => new FifoStretchBufferAdapter();
var createCircularStretchInputBufferAdapter = () => new CircularStretchInputBufferAdapter();
var USE_AUTO_SEQUENCE_LEN = 0;
var DEFAULT_SEQUENCE_MS = USE_AUTO_SEQUENCE_LEN;
var USE_AUTO_SEEKWINDOW_LEN = 0;
var DEFAULT_SEEKWINDOW_MS = USE_AUTO_SEEKWINDOW_LEN;
var DEFAULT_OVERLAP_MS = 8;
var AUTOSEQ_TEMPO_LOW = 0.25;
var AUTOSEQ_TEMPO_TOP = 4;
var AUTOSEQ_AT_MIN = 125;
var AUTOSEQ_AT_MAX = 50;
var AUTOSEQ_K = (AUTOSEQ_AT_MAX - AUTOSEQ_AT_MIN) / (AUTOSEQ_TEMPO_TOP - AUTOSEQ_TEMPO_LOW);
var AUTOSEQ_C = AUTOSEQ_AT_MIN - AUTOSEQ_K * AUTOSEQ_TEMPO_LOW;
var AUTOSEEK_AT_MIN = 25;
var AUTOSEEK_AT_MAX = 15;
var AUTOSEEK_K = (AUTOSEEK_AT_MAX - AUTOSEEK_AT_MIN) / (AUTOSEQ_TEMPO_TOP - AUTOSEQ_TEMPO_LOW);
var AUTOSEEK_C = AUTOSEEK_AT_MIN - AUTOSEEK_K * AUTOSEQ_TEMPO_LOW;
var NORMALIZED_CORRELATION_EPSILON = 1e-12;
var QUICK_SEEK_FALLBACK_THRESHOLD = 256;
var QUICK_SEEK_MIN_VALID_CANDIDATES = 8;
var Stretch = class _Stretch extends AbstractSamplePipe {
  /**
   * Creates a Stretch instance.
   * @param options Constructor options.
   */
  constructor({ sampleRate = 44100, createBuffers = false, inputBufferAdapterFactory = createFifoStretchInputBufferAdapter, sampleBufferFactory = () => new FifoSampleBuffer() } = {}) {
    super({
      createBuffers,
      inputBufferFactory: sampleBufferFactory,
      outputBufferFactory: sampleBufferFactory
    });
    __publicField(this, "inputBufferAdapterFactory");
    __publicField(this, "sampleBufferFactory");
    __publicField(this, "inputBufferAdapter");
    __publicField(this, "outputBufferAdapter");
    __publicField(this, "overlapScratch");
    __publicField(this, "_quickSeek");
    __publicField(this, "midBufferDirty");
    __publicField(this, "midBuffer");
    __publicField(this, "refMidBuffer");
    __publicField(this, "refMidBufferEnergy");
    __publicField(this, "overlapLength");
    __publicField(this, "autoSeqSetting");
    __publicField(this, "autoSeekSetting");
    __publicField(this, "_tempo");
    __publicField(this, "sampleRate");
    __publicField(this, "_overlapMs");
    __publicField(this, "sequenceMs");
    __publicField(this, "seekWindowMs");
    __publicField(this, "seekWindowLength");
    __publicField(this, "seekLength");
    __publicField(this, "nominalSkip");
    __publicField(this, "skipFract");
    __publicField(this, "sampleReq");
    this.inputBufferAdapterFactory = inputBufferAdapterFactory;
    this.sampleBufferFactory = sampleBufferFactory;
    this.inputBufferAdapter = inputBufferAdapterFactory();
    this.outputBufferAdapter = new GenericStretchWriteBufferAdapter();
    this.overlapScratch = new Float32Array(0);
    this._quickSeek = true;
    this.midBufferDirty = true;
    this.midBuffer = null;
    this.refMidBufferEnergy = 0;
    this.overlapLength = 0;
    this.autoSeqSetting = true;
    this.autoSeekSetting = true;
    this._tempo = 1;
    this.setParameters(sampleRate, DEFAULT_SEQUENCE_MS, DEFAULT_SEEKWINDOW_MS, DEFAULT_OVERLAP_MS);
  }
  clear() {
    super.clear();
    this.clearMidBuffer();
  }
  clearMidBuffer() {
    this.midBufferDirty = true;
    if (this.midBuffer) {
      this.midBuffer.fill(0);
    }
    if (this.refMidBuffer) {
      this.refMidBuffer.fill(0);
    }
    this.skipFract = 0;
  }
  setParameters(sampleRate, sequenceMs, seekWindowMs, overlapMs) {
    if (sampleRate > 0) {
      this.sampleRate = sampleRate;
    }
    if (overlapMs > 0) {
      this._overlapMs = overlapMs;
    }
    if (sequenceMs > 0) {
      this.sequenceMs = sequenceMs;
      this.autoSeqSetting = false;
    } else {
      this.autoSeqSetting = true;
    }
    if (seekWindowMs > 0) {
      this.seekWindowMs = seekWindowMs;
      this.autoSeekSetting = false;
    } else {
      this.autoSeekSetting = true;
    }
    this.calculateSequenceParameters();
    this.calculateOverlapLength(this._overlapMs);
    this.updateTempoDerivedState();
  }
  set tempo(newTempo) {
    this._tempo = newTempo;
    this.updateTempoDerivedState();
  }
  get tempo() {
    return this._tempo;
  }
  get inputChunkSize() {
    return this.sampleReq;
  }
  get outputChunkSize() {
    return this.overlapLength + Math.max(0, this.seekWindowLength - 2 * this.overlapLength);
  }
  calculateOverlapLength(overlapInMsec = 0) {
    let newOvl = this.sampleRate * overlapInMsec / 1e3;
    newOvl = newOvl < 16 ? 16 : newOvl;
    newOvl -= newOvl % 8;
    if (newOvl === this.overlapLength && this.midBuffer !== null) {
      return;
    }
    this.overlapLength = newOvl;
    const needed = this.overlapLength * 2;
    if (!this.refMidBuffer || this.refMidBuffer.length < needed) {
      this.refMidBuffer = new Float32Array(needed);
    }
    if (!this.midBuffer || this.midBuffer.length < needed) {
      this.midBuffer = new Float32Array(needed);
    }
  }
  checkLimits(x, mi, ma) {
    return x < mi ? mi : x > ma ? ma : x;
  }
  calculateSequenceParameters() {
    if (this.autoSeqSetting) {
      let seq = AUTOSEQ_C + AUTOSEQ_K * this._tempo;
      seq = this.checkLimits(seq, AUTOSEQ_AT_MAX, AUTOSEQ_AT_MIN);
      this.sequenceMs = Math.floor(seq + 0.5);
    }
    if (this.autoSeekSetting) {
      let seek = AUTOSEEK_C + AUTOSEEK_K * this._tempo;
      seek = this.checkLimits(seek, AUTOSEEK_AT_MAX, AUTOSEEK_AT_MIN);
      this.seekWindowMs = Math.floor(seek + 0.5);
    }
    this.seekWindowLength = Math.floor(this.sampleRate * this.sequenceMs / 1e3);
    this.seekLength = Math.floor(this.sampleRate * this.seekWindowMs / 1e3);
    this.normalizeWindowInvariants();
  }
  normalizeWindowInvariants() {
    this.seekLength = Math.max(1, this.seekLength);
    this.seekWindowLength = Math.max(this.seekWindowLength, this.overlapLength);
  }
  updateTempoDerivedState() {
    this.calculateSequenceParameters();
    this.nominalSkip = this._tempo * (this.seekWindowLength - this.overlapLength);
    this.skipFract = 0;
    const intskip = Math.floor(this.nominalSkip + 0.5);
    this.sampleReq = Math.max(intskip + this.overlapLength, this.seekWindowLength) + this.seekLength;
  }
  /**
   * Whether the fast multi-pass seek algorithm is active.
   * @returns `true` if quick seek is enabled (default); `false` for exhaustive search.
   */
  get quickSeek() {
    return this._quickSeek;
  }
  set quickSeek(enable) {
    this._quickSeek = enable;
  }
  /**
   * Current overlap crossfade length in milliseconds.
   * @returns The overlap period used at the current sample rate.
   */
  get overlapMs() {
    return this._overlapMs;
  }
  /**
   * Sets the overlap crossfade length and recalculates derived parameters.
   * @param ms Overlap period in milliseconds (must be > 0).
   */
  set overlapMs(ms) {
    if (ms > 0) {
      this._overlapMs = ms;
      this.calculateOverlapLength(this._overlapMs);
      this.calculateSequenceParameters();
      this.updateTempoDerivedState();
    }
  }
  /**
   * Applies a partial set of WSOLA timing parameters.
   *
   * @remarks
   * Only the provided fields are updated; omitted fields remain unchanged.
   * Pass `sequenceMs: 0` or `seekWindowMs: 0` to switch that dimension back to auto-calculation.
   *
   * @param params Partial set of WSOLA timing parameters to apply.
   *
   * @example
   * stretch.setStretchParameters({ overlapMs: 12, quickSeek: false });
   */
  setStretchParameters(params) {
    if (params.quickSeek !== void 0) {
      this._quickSeek = params.quickSeek;
    }
    let needsRecalc = false;
    if (params.sequenceMs !== void 0) {
      if (params.sequenceMs > 0) {
        this.sequenceMs = params.sequenceMs;
        this.autoSeqSetting = false;
      } else {
        this.autoSeqSetting = true;
      }
      needsRecalc = true;
    }
    if (params.seekWindowMs !== void 0) {
      if (params.seekWindowMs > 0) {
        this.seekWindowMs = params.seekWindowMs;
        this.autoSeekSetting = false;
      } else {
        this.autoSeekSetting = true;
      }
      needsRecalc = true;
    }
    if (params.overlapMs !== void 0 && params.overlapMs > 0) {
      this._overlapMs = params.overlapMs;
      this.calculateOverlapLength(this._overlapMs);
      needsRecalc = true;
    }
    if (needsRecalc) {
      this.calculateSequenceParameters();
      this.updateTempoDerivedState();
    }
  }
  clone() {
    const result = new _Stretch({
      createBuffers: false,
      inputBufferAdapterFactory: this.inputBufferAdapterFactory,
      sampleBufferFactory: this.sampleBufferFactory
    });
    result.tempo = this._tempo;
    result.setParameters(this.sampleRate, this.sequenceMs, this.seekWindowMs, this._overlapMs);
    return result;
  }
  seekBestOverlapPosition(inputBuffer) {
    const resolvedInputBuffer = inputBuffer ?? this.getInputBufferAdapter();
    if (!this._quickSeek || this.seekLength <= QUICK_SEEK_FALLBACK_THRESHOLD) {
      return this.seekBestOverlapPositionStereo(resolvedInputBuffer);
    }
    return this.seekBestOverlapPositionStereoQuick(resolvedInputBuffer);
  }
  seekBestOverlapPositionStereo(inputBuffer) {
    let bestOffset;
    let bestCorrelation;
    let correlation;
    this.preCalculateCorrelationReferenceStereo();
    bestOffset = 0;
    bestCorrelation = -Infinity;
    for (let i = 0; i < this.seekLength; i++) {
      correlation = this.calculateCrossCorrelationStereo(2 * i, this.refMidBuffer, inputBuffer);
      if (correlation > bestCorrelation) {
        bestCorrelation = correlation;
        bestOffset = i;
      }
    }
    return bestOffset;
  }
  seekBestOverlapPositionStereoQuick(inputBuffer) {
    let bestOffset;
    let bestCorrelation;
    let correlation;
    let correlationOffset;
    let tempOffset;
    let evaluatedCandidates;
    this.preCalculateCorrelationReferenceStereo();
    bestCorrelation = this.calculateCrossCorrelationStereo(0, this.refMidBuffer, inputBuffer);
    evaluatedCandidates = 1;
    bestOffset = 0;
    correlationOffset = 0;
    for (let scanCount = 0; scanCount < 4; scanCount++) {
      let previousTempOffset = Number.MIN_SAFE_INTEGER;
      const scanOffsets = this.getQuickScanOffsets(scanCount);
      for (const scanOffset of scanOffsets) {
        tempOffset = correlationOffset + scanOffset;
        if (tempOffset === previousTempOffset) {
          continue;
        }
        previousTempOffset = tempOffset;
        if (tempOffset < 0) {
          continue;
        }
        if (tempOffset >= this.seekLength) {
          continue;
        }
        correlation = this.calculateCrossCorrelationStereo(2 * tempOffset, this.refMidBuffer, inputBuffer);
        evaluatedCandidates++;
        if (correlation > bestCorrelation) {
          bestCorrelation = correlation;
          bestOffset = tempOffset;
        }
      }
      correlationOffset = bestOffset;
    }
    if (evaluatedCandidates < QUICK_SEEK_MIN_VALID_CANDIDATES) {
      return this.seekBestOverlapPositionStereo(inputBuffer);
    }
    return bestOffset;
  }
  getQuickScanOffsets(stage) {
    const maxOffset = Math.max(1, this.seekLength - 1);
    if (stage === 0) {
      return this.generateFractionalScanOffsets(maxOffset, 2, 1, 14, 24);
    }
    if (stage === 1) {
      return this.generateSymmetricScanOffsets(maxOffset, 0.2);
    }
    if (stage === 2) {
      return this.generateSymmetricScanOffsets(maxOffset, 0.06);
    }
    return this.generateSymmetricScanOffsets(maxOffset, 0.015);
  }
  generateFractionalScanOffsets(maxOffset, startNumerator, stepNumerator, denominator, steps) {
    const offsets = [];
    const seen = /* @__PURE__ */ new Set();
    const safeDenominator = Math.max(1, denominator);
    const safeSteps = Math.max(1, steps);
    for (let i = 0; i < safeSteps; i++) {
      const numerator = startNumerator + i * stepNumerator;
      const value = Math.round(maxOffset * numerator / safeDenominator);
      if (value <= 0 || value >= this.seekLength || seen.has(value)) {
        continue;
      }
      seen.add(value);
      offsets.push(value);
    }
    return offsets;
  }
  generateSymmetricScanOffsets(maxOffset, spanRatio) {
    const span = Math.max(1, Math.round(maxOffset * spanRatio));
    const scales = [1, 0.75, 0.5, 0.25];
    const negative = [];
    const positive = [];
    const seen = /* @__PURE__ */ new Set();
    for (const scale of scales) {
      const magnitude = Math.max(1, Math.round(span * scale));
      const neg = -magnitude;
      const pos = magnitude;
      if (!seen.has(neg)) {
        seen.add(neg);
        negative.push(neg);
      }
      if (!seen.has(pos)) {
        seen.add(pos);
        positive.push(pos);
      }
    }
    return negative.concat(positive);
  }
  preCalculateCorrelationReferenceStereo() {
    let energy = 0;
    for (let i = 0; i < this.overlapLength; i++) {
      const temp = i * (this.overlapLength - i);
      const ctx = i * 2;
      const left = this.midBuffer[ctx] * temp;
      const right = this.midBuffer[ctx + 1] * temp;
      this.refMidBuffer[ctx] = left;
      this.refMidBuffer[ctx + 1] = right;
      energy += left * left + right * right;
    }
    this.refMidBufferEnergy = energy;
  }
  calculateCrossCorrelationStereo(mixingPos, compare, inputBuffer) {
    mixingPos += inputBuffer.startIndex;
    let dot = 0;
    let sourceEnergy = 0;
    const calcLength = 2 * this.overlapLength;
    const source = inputBuffer.readSubarray(mixingPos, mixingPos + calcLength);
    for (let i = 0; i < calcLength; i += 2) {
      const sourceLeft = i < source.length ? source[i] : 0;
      const sourceRight = i + 1 < source.length ? source[i + 1] : 0;
      const compareLeft = compare[i];
      const compareRight = compare[i + 1];
      dot += sourceLeft * compareLeft + sourceRight * compareRight;
      sourceEnergy += sourceLeft * sourceLeft + sourceRight * sourceRight;
    }
    if (sourceEnergy <= NORMALIZED_CORRELATION_EPSILON || this.refMidBufferEnergy <= NORMALIZED_CORRELATION_EPSILON) {
      return -1;
    }
    return dot / Math.sqrt(sourceEnergy * this.refMidBufferEnergy);
  }
  overlapStereo(inputPosition, inputBuffer, outputBuffer) {
    inputPosition += inputBuffer.startIndex;
    const overlapSamples = this.overlapLength * 2;
    if (this.overlapScratch.length < overlapSamples) {
      this.overlapScratch = new Float32Array(overlapSamples);
    }
    const output = this.overlapScratch;
    const input = inputBuffer.readSubarray(inputPosition, inputPosition + overlapSamples);
    const frameScale = 1 / this.overlapLength;
    for (let i = 0; i < this.overlapLength; i++) {
      const tempFrame = (this.overlapLength - i) * frameScale;
      const fi = i * frameScale;
      const ctx = 2 * i;
      const inputLeft = ctx < input.length ? input[ctx] : 0;
      const inputRight = ctx + 1 < input.length ? input[ctx + 1] : 0;
      output[ctx] = inputLeft * fi + this.midBuffer[ctx] * tempFrame;
      output[ctx + 1] = inputRight * fi + this.midBuffer[ctx + 1] * tempFrame;
    }
    outputBuffer.appendSamples(output, this.overlapLength);
  }
  process() {
    const inputBuffer = this.getInputBufferAdapter();
    const outputBuffer = this.getOutputBufferAdapter();
    if (!this.bootstrapMidBuffer(inputBuffer)) {
      return;
    }
    while (inputBuffer.frameCount >= this.sampleReq) {
      this.processOneWindow(inputBuffer, outputBuffer);
    }
  }
  bootstrapMidBuffer(inputBuffer) {
    if (!this.midBufferDirty) {
      return true;
    }
    if (inputBuffer.frameCount < this.overlapLength) {
      return false;
    }
    const needed = this.overlapLength * 2;
    if (!this.midBuffer || this.midBuffer.length < needed) {
      this.midBuffer = new Float32Array(needed);
    }
    inputBuffer.receiveSamples(this.midBuffer, this.overlapLength);
    this.midBufferDirty = false;
    return true;
  }
  processOneWindow(inputBuffer, outputBuffer) {
    const offset = this.seekBestOverlapPosition(inputBuffer);
    this.overlapStereo(2 * Math.floor(offset), inputBuffer, outputBuffer);
    const middleFrames = this.seekWindowLength - 2 * this.overlapLength;
    if (middleFrames > 0) {
      outputBuffer.putFrom(inputBuffer, offset + this.overlapLength, middleFrames);
    }
    this.captureOverlapHistory(offset, inputBuffer);
    this.advanceInputByNominalSkip(inputBuffer);
  }
  captureOverlapHistory(offset, inputBuffer) {
    const start = inputBuffer.startIndex + 2 * (offset + this.seekWindowLength - this.overlapLength);
    this.midBuffer.set(inputBuffer.readSubarray(start, start + 2 * this.overlapLength));
  }
  advanceInputByNominalSkip(inputBuffer) {
    this.skipFract += this.nominalSkip;
    const overlapSkip = Math.floor(this.skipFract);
    this.skipFract -= overlapSkip;
    inputBuffer.receive(overlapSkip);
  }
  getInputBufferAdapter() {
    if (this._inputBuffer === null) {
      throw new Error("inputBuffer is not set");
    }
    this.inputBufferAdapter.setBuffer(this._inputBuffer);
    return this.inputBufferAdapter;
  }
  getOutputBufferAdapter() {
    if (this._outputBuffer === null) {
      throw new Error("outputBuffer is not set");
    }
    this.outputBufferAdapter.setOutputBuffer(this._outputBuffer);
    return this.outputBufferAdapter;
  }
};

// node_modules/@soundtouchjs/core/.dist/testFloatEqual.js
function isFloatDifferent(a, b) {
  return (a > b ? a - b : b - a) > 1e-10;
}

// node_modules/@soundtouchjs/core/.dist/SoundTouch.js
var SoundTouch = class _SoundTouch {
  /**
   * Creates a new SoundTouch processor instance.
   * @param options Construction options for sample rate, buffer strategy, and factories.
   */
  constructor(options = {}) {
    __publicField(this, "transposer");
    __publicField(this, "stretch");
    __publicField(this, "_sampleRate");
    __publicField(this, "_sampleBufferType");
    __publicField(this, "_sampleBufferFactory");
    __publicField(this, "_interpolationStrategy");
    __publicField(this, "_inputBuffer");
    __publicField(this, "_intermediateBuffer");
    __publicField(this, "_outputBuffer");
    __publicField(this, "_rate");
    __publicField(this, "_tempo");
    /** Current pitch multiplier. Updated by the `pitch`, `pitchOctaves`, and `pitchSemitones` setters. */
    __publicField(this, "virtualPitch");
    this._sampleBufferType = options.sampleBufferType ?? "circular";
    this._sampleBufferFactory = options.sampleBufferFactory ?? (this._sampleBufferType === "fifo" ? () => new FifoSampleBuffer() : () => new CircularSampleBuffer());
    this.transposer = new RateTransposer({
      createBuffers: false,
      sampleBufferAdapterFactory: this._sampleBufferType === "circular" ? createCircularSampleBufferAdapter : createFifoSampleBufferAdapter,
      sampleBufferFactory: this._sampleBufferFactory,
      interpolationStrategy: options.interpolationStrategy
    });
    this._interpolationStrategy = this.transposer.strategy;
    this._sampleRate = options.sampleRate ?? 44100;
    if (options.stretchFactory) {
      this.stretch = options.stretchFactory(this._sampleRate, {
        sampleBufferFactory: this._sampleBufferFactory,
        sampleBufferType: this._sampleBufferType
      });
    } else {
      this.stretch = new Stretch({
        sampleRate: this._sampleRate,
        createBuffers: false,
        inputBufferAdapterFactory: this._sampleBufferType === "circular" ? createCircularStretchInputBufferAdapter : createFifoStretchInputBufferAdapter,
        sampleBufferFactory: this._sampleBufferFactory
      });
    }
    this.stretch.setParameters(this._sampleRate, 0, 0, 0);
    this._inputBuffer = this._sampleBufferFactory();
    this._intermediateBuffer = this._sampleBufferFactory();
    this._outputBuffer = this._sampleBufferFactory();
    this._rate = 0;
    this._tempo = 0;
    this.virtualPitch = 1;
    this.calculateEffectiveRateAndTempo();
  }
  /**
   * Clears both processing stages and their internal buffers.
   * @remarks
   * Resets the state of the transposer and stretch stages, including all internal buffers.
   */
  clear() {
    this.transposer.clear();
    this.stretch.clear();
  }
  /**
   * Creates an independent copy with equivalent runtime configuration.
   */
  clone() {
    const result = new _SoundTouch({
      sampleRate: this._sampleRate,
      sampleBufferType: this._sampleBufferType,
      sampleBufferFactory: this._sampleBufferFactory,
      interpolationStrategy: {
        id: this._interpolationStrategy,
        params: this.transposer.strategyParams
      }
    });
    result.pitch = this.virtualPitch;
    return result;
  }
  /**
   * Active interpolation strategy id used by the transposer stage.
   * @returns The current interpolation strategy identifier.
   */
  get interpolationStrategy() {
    return this._interpolationStrategy;
  }
  /**
   * Active interpolation strategy params used by the transposer stage.
   * @returns The current interpolation strategy parameters.
   */
  get interpolationStrategyParams() {
    return this.transposer.strategyParams;
  }
  /**
   * Switches interpolation strategy at runtime.
   * @param strategy The new interpolation strategy to use.
   */
  setInterpolationStrategy(strategy) {
    this.transposer.setInterpolationStrategy(strategy);
    this._interpolationStrategy = this.transposer.strategy;
  }
  /**
   * Applies a partial runtime params update to the current strategy.
   * @param params Partial set of parameters to update.
   */
  setInterpolationStrategyParams(params) {
    this.transposer.setInterpolationStrategyParams(params);
  }
  /**
   * Applies a partial set of WSOLA timing parameters to the stretch stage.
   *
   * @remarks
   * Delegates directly to {@link Stretch.setStretchParameters}. Only the provided
   * fields are updated; omitted fields remain unchanged. Pass `sequenceMs: 0` or
   * `seekWindowMs: 0` to switch that dimension back to auto-calculation.
   *
   * @param params Partial set of WSOLA timing parameters to apply.
   *
   * @example
   * st.setStretchParameters({ overlapMs: 12, quickSeek: false });
   */
  setStretchParameters(params) {
    this.stretch.setStretchParameters(params);
  }
  /**
   * Sets the pitch multiplier and recomputes the derived pipeline rate and tempo.
   *
   * @remarks
   * Internally sets `_rate = pitch` and `_tempo = 1 / pitch`, rewiring the
   * Transposer→Stretch stage order when pitch > 1.
   */
  set pitch(pitch) {
    this.virtualPitch = pitch;
    this.calculateEffectiveRateAndTempo();
  }
  /**
   * Sets pitch by octave offset.
   */
  set pitchOctaves(pitchOctaves) {
    this.pitch = Math.exp(0.69314718056 * pitchOctaves);
    this.calculateEffectiveRateAndTempo();
  }
  /**
   * Sets pitch by semitone offset.
   */
  set pitchSemitones(pitchSemitones) {
    this.pitchOctaves = pitchSemitones / 12;
  }
  /**
   * Input buffer for upstream interleaved stereo frames.
   * @returns The input buffer for writing audio frames.
   */
  get inputBuffer() {
    return this._inputBuffer;
  }
  /**
   * Output buffer that downstream consumers read from.
   * @returns The output buffer for reading processed audio frames.
   */
  get outputBuffer() {
    return this._outputBuffer;
  }
  /**
   * Recomputes the effective pipeline rate/tempo from `virtualPitch` and rewires stage order when needed.
   *
   * @remarks
   * `_rate` is set to `virtualPitch`; `_tempo` to `1 / virtualPitch`. When `_rate > 1` the
   * Stretch stage feeds the Transposer; otherwise the order is reversed.
   */
  calculateEffectiveRateAndTempo() {
    const previousTempo = this._tempo;
    const previousRate = this._rate;
    this._tempo = 1 / this.virtualPitch;
    this._rate = this.virtualPitch;
    if (isFloatDifferent(this._tempo, previousTempo)) {
      this.stretch.tempo = this._tempo;
    }
    if (isFloatDifferent(this._rate, previousRate)) {
      this.transposer.rate = this._rate;
    }
    if (this._rate > 1) {
      if (this._outputBuffer !== this.transposer.outputBuffer) {
        this.stretch.inputBuffer = this._inputBuffer;
        this.stretch.outputBuffer = this._intermediateBuffer;
        this.transposer.inputBuffer = this._intermediateBuffer;
        this.transposer.outputBuffer = this._outputBuffer;
      }
    } else {
      if (this._outputBuffer !== this.stretch.outputBuffer) {
        this.transposer.inputBuffer = this._inputBuffer;
        this.transposer.outputBuffer = this._intermediateBuffer;
        this.stretch.inputBuffer = this._intermediateBuffer;
        this.stretch.outputBuffer = this._outputBuffer;
      }
    }
  }
  /**
   * Runs one processing step through the currently selected stage order.
   * @remarks
   * Processes available frames through the pipeline, updating output buffers.
   */
  process() {
    if (this._rate > 1) {
      this.stretch.process();
      this.transposer.process();
    } else {
      this.transposer.process();
      this.stretch.process();
    }
  }
};

// audio-transform.js
function resample(input, length) {
  const output = new Float32Array(length), scale = (input.length - 1) / Math.max(1, length - 1);
  for (let i = 0; i < length; i++) {
    const position = i * scale, left = Math.floor(position), fraction = position - left;
    output[i] = input[left] * (1 - fraction) + input[Math.min(left + 1, input.length - 1)] * fraction;
  }
  return output;
}
function transformNote(input, sampleRate, semitones, stretch) {
  if (!input.length) throw new Error("Cannot edit an empty note.");
  if (!Number.isFinite(stretch) || stretch <= 0 || !Number.isFinite(semitones)) throw new Error("Invalid note edit.");
  if (stretch === 1 && semitones === 0) return input.slice();
  const targetLength = Math.max(1, Math.round(input.length * stretch));
  const slowed = resample(input, targetLength), processor = new SoundTouch({ sampleRate, sampleBufferType: "fifo" });
  processor.pitch = stretch * 2 ** (semitones / 12);
  const chunks = [], chunkSize = 2048;
  let outputFrames = 0;
  function feed(data, first, count) {
    const stereo = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) stereo[2 * i] = stereo[2 * i + 1] = data[first + i];
    processor.inputBuffer.putSamples(stereo, 0, count);
    processor.process();
    const frames = processor.outputBuffer.frameCount;
    if (frames) {
      const returned = new Float32Array(frames * 2);
      processor.outputBuffer.receiveSamples(returned, frames);
      const mono = new Float32Array(frames);
      for (let i = 0; i < frames; i++) mono[i] = returned[2 * i];
      chunks.push(mono);
      outputFrames += frames;
    }
  }
  for (let i = 0; i < slowed.length; i += chunkSize) feed(slowed, i, Math.min(chunkSize, slowed.length - i));
  const silence = new Float32Array(chunkSize);
  for (let pass = 0; pass < 10 && outputFrames < targetLength; pass++) feed(silence, 0, silence.length);
  const output = new Float32Array(targetLength);
  let cursor = 0;
  for (const chunk of chunks) {
    if (cursor >= targetLength) break;
    const count = Math.min(chunk.length, targetLength - cursor);
    output.set(chunk.subarray(0, count), cursor);
    cursor += count;
  }
  return output;
}
export {
  transformNote
};
