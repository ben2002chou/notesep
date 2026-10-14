// Float WAV export preserves note levels and the direct sum without clipping.
export function floatWav(samples, sampleRate) {
  const bytes = new ArrayBuffer(44 + samples.length * 4);
  const view = new DataView(bytes);
  const word = (offset, value) => {
    for (let index = 0; index < value.length; index++) view.setUint8(offset + index, value.charCodeAt(index));
  };
  word(0, 'RIFF'); view.setUint32(4, bytes.byteLength - 8, true); word(8, 'WAVE');
  word(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 3, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true); view.setUint16(32, 4, true);
  view.setUint16(34, 32, true); word(36, 'data'); view.setUint32(40, samples.length * 4, true);
  for (let index = 0; index < samples.length; index++) {
    if (!Number.isFinite(samples[index])) throw new Error('Non-finite rendered audio');
    view.setFloat32(44 + index * 4, samples[index], true);
  }
  return bytes;
}

export function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
