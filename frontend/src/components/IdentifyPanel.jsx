import { useState, useRef, useEffect } from 'react'
import CameraCapture from './CameraCapture.jsx'

/*
 * Measured false-accept / false-reject rates for the dlib engine, from 3,000
 * impostor and 3,000 genuine LFW pairs on identities the model never saw
 * (python -m ml.calibrate). [threshold, FAR, FRR].
 *
 * This is here because a bare number is not a decision: 0.79 looks like a
 * small nudge from 0.60 but accepts 40% of strangers instead of 0.6%. The
 * slider now states the consequence rather than leaving it to be discovered.
 */
const ERROR_CURVE = [
  [0.30, 0.0000, 0.9533], [0.35, 0.0000, 0.8383], [0.40, 0.0000, 0.6357],
  [0.45, 0.0000, 0.4130], [0.50, 0.0000, 0.2317], [0.55, 0.0003, 0.1213],
  [0.60, 0.0063, 0.0517], [0.65, 0.0330, 0.0233], [0.70, 0.0897, 0.0130],
  [0.75, 0.2217, 0.0063], [0.80, 0.4053, 0.0033], [0.85, 0.6057, 0.0000],
  [0.90, 0.7937, 0.0000],
]

const RECOMMENDED = 0.60

function ratesAt(threshold) {
  const first = ERROR_CURVE[0]
  const last = ERROR_CURVE[ERROR_CURVE.length - 1]
  if (threshold <= first[0]) return { far: first[1], frr: first[2] }
  if (threshold >= last[0]) return { far: last[1], frr: last[2] }
  for (let i = 0; i < ERROR_CURVE.length - 1; i++) {
    const [t0, far0, frr0] = ERROR_CURVE[i]
    const [t1, far1, frr1] = ERROR_CURVE[i + 1]
    if (threshold >= t0 && threshold <= t1) {
      const k = (threshold - t0) / (t1 - t0)
      return { far: far0 + k * (far1 - far0), frr: frr0 + k * (frr1 - frr0) }
    }
  }
  return { far: last[1], frr: last[2] }
}

function describeRisk(threshold) {
  const { far, frr } = ratesAt(threshold)
  // "1 in N" stops being honest once N approaches 1: at FAR 0.79 it rounds to
  // "1 in 1". Past a half, say it straight.
  let strangers
  if (far < 0.001) strangers = 'almost never matches a stranger'
  else if (far >= 0.5) strangers = `most strangers will match (${Math.round(far * 100)}%)`
  else if (far >= 0.25) strangers = `about ${Math.round(far * 100)}% of strangers will match`
  else strangers = `about 1 in ${Math.round(1 / far)} strangers may match`
  const missed = `misses about ${Math.round(frr * 100)}% of real matches`
  let tone = 'ok'
  if (far >= 0.20) tone = 'danger'
  else if (far >= 0.03) tone = 'warn'
  else if (frr >= 0.25) tone = 'warn'
  return { text: `${strangers} · ${missed}`, tone }
}

export default function IdentifyPanel({ onIdentified, onError }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [threshold, setThreshold] = useState(0.6)
  const [dragging, setDragging] = useState(false)
  const [running, setRunning] = useState(false)
  const [inlineError, setInlineError] = useState(null)
  const [cameraOpen, setCameraOpen] = useState(false)
  const inputRef = useRef(null)
  const risk = describeRisk(threshold)

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const acceptFile = (candidate) => {
    if (!candidate) return
    if (!candidate.type.startsWith('image/')) {
      setInlineError('That file is not an image. Choose a JPG, PNG, BMP or WebP photo.')
      return
    }
    setInlineError(null)
    setFile(candidate)
  }

  const handleIdentify = async () => {
    if (!file) {
      setInlineError('Choose a photo first.')
      return
    }

    setInlineError(null)
    setRunning(true)

    const body = new FormData()
    body.append('image', file)
    body.append('threshold', threshold.toFixed(2))

    try {
      const res = await fetch('/identify', {
        method: 'POST',
        headers: { Accept: 'application/json' },
        body,
      })
      const data = await res.json().catch(() => null)

      if (!res.ok || !data || data.error) {
        const message = data?.error ?? 'Identifying failed. Check that the server is running.'
        setInlineError(message)
        onError(message)
        return
      }

      onIdentified(data)
    } catch (err) {
      const message = `Could not reach the server: ${err.message}`
      setInlineError(message)
      onError(message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <section className="card">
      <div className="card-head">
        <h2 className="card-title">Identify a face</h2>
      </div>

      <div className="field">
        <label className="field-label" htmlFor="identify-file">Photo</label>
        {cameraOpen ? (
          <CameraCapture
            disabled={running}
            onCapture={file => { setCameraOpen(false); acceptFile(file) }}
            onError={setInlineError}
            onClose={() => setCameraOpen(false)}
          />
        ) : (
        <button
          type="button"
          id="identify-file"
          className={`upload${dragging ? ' is-dragging' : ''}`}
          disabled={running}
          onClick={() => inputRef.current?.click()}
          onDragOver={event => { event.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={event => {
            event.preventDefault()
            setDragging(false)
            if (running) return
            acceptFile(event.dataTransfer.files[0])
          }}
        >
          {file ? (
            <>
              <div className="upload-main">{file.name}</div>
              <div className="upload-hint">Click to choose a different photo</div>
            </>
          ) : (
            <>
              <div className="upload-arrow" aria-hidden="true">&#8593;</div>
              <div className="upload-main">Drop a photo here, or click to choose</div>
              <div className="upload-hint">JPG, PNG, BMP or WebP. Up to 16 MB.</div>
            </>
          )}
        </button>
        )}

        {!cameraOpen && (
          <button
            type="button"
            className="btn-link"
            onClick={() => { setInlineError(null); setCameraOpen(true) }}
            disabled={running}
          >
            Use camera instead
          </button>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={event => acceptFile(event.target.files[0])}
        />

        {file && preview && <img className="preview" src={preview} alt={file.name} />}
      </div>

      <div className="field">
        <label className="field-label" htmlFor="identify-threshold">
          Match strictness
        </label>
        <div className="threshold">
          <input
            id="identify-threshold"
            className="range"
            type="range"
            min="0.30"
            max="0.90"
            step="0.01"
            value={threshold}
            disabled={running}
            onChange={event => setThreshold(parseFloat(event.target.value))}
          />
          <span className="threshold-value">{threshold.toFixed(2)}</span>
        </div>

        <p className={`risk risk-${risk.tone}`}>{risk.text}</p>

        {Math.abs(threshold - RECOMMENDED) > 0.005 && (
          <button
            type="button"
            className="btn-link"
            onClick={() => setThreshold(RECOMMENDED)}
            disabled={running}
          >
            Reset to the recommended {RECOMMENDED.toFixed(2)}
          </button>
        )}
      </div>

      {inlineError && <p className="error-text">{inlineError}</p>}

      <button
        type="button"
        className="btn btn-block"
        onClick={handleIdentify}
        disabled={running || !file}
      >
        {running ? (
          <>
            <span className="spinner" aria-hidden="true" />
            Identifying
          </>
        ) : (
          'Identify'
        )}
      </button>
    </section>
  )
}
