import { useState, useRef, useEffect } from 'react'
import CameraCapture from '../components/CameraCapture.jsx'
import { ROUTES, navigate } from '../router.js'

/*
 * Measured false-accept / false-reject rates for the dlib engine, from 3,000
 * impostor and 3,000 genuine LFW pairs on identities the model never saw
 * (python -m ml.calibrate). [threshold, FAR, FRR].
 */
const ERROR_CURVE = [
  [0.30, 0.0000, 0.9533], [0.35, 0.0000, 0.8383], [0.40, 0.0000, 0.6357],
  [0.45, 0.0000, 0.4130], [0.50, 0.0000, 0.2317], [0.55, 0.0003, 0.1213],
  [0.60, 0.0063, 0.0517], [0.65, 0.0330, 0.0233], [0.70, 0.0897, 0.0130],
  [0.75, 0.2217, 0.0063], [0.80, 0.4053, 0.0033], [0.85, 0.6057, 0.0000],
  [0.90, 0.7937, 0.0000],
]
const RECOMMENDED = 0.60

function ratesAt(t) {
  const first = ERROR_CURVE[0], last = ERROR_CURVE[ERROR_CURVE.length - 1]
  if (t <= first[0]) return { far: first[1], frr: first[2] }
  if (t >= last[0]) return { far: last[1], frr: last[2] }
  for (let i = 0; i < ERROR_CURVE.length - 1; i++) {
    const [t0, f0, r0] = ERROR_CURVE[i], [t1, f1, r1] = ERROR_CURVE[i + 1]
    if (t >= t0 && t <= t1) {
      const k = (t - t0) / (t1 - t0)
      return { far: f0 + k * (f1 - f0), frr: r0 + k * (r1 - r0) }
    }
  }
  return { far: last[1], frr: last[2] }
}

function describeRisk(t) {
  const { far, frr } = ratesAt(t)
  let strangers
  if (far < 0.001) strangers = 'almost never matches a stranger'
  else if (far >= 0.5) strangers = `most strangers will match (${Math.round(far * 100)}%)`
  else if (far >= 0.25) strangers = `about ${Math.round(far * 100)}% of strangers will match`
  else strangers = `about 1 in ${Math.round(1 / far)} strangers may match`
  let tone = 'ok'
  if (far >= 0.20) tone = 'danger'
  else if (far >= 0.03 || frr >= 0.25) tone = 'warn'
  return { text: `${strangers} · misses about ${Math.round(frr * 100)}% of real matches`, tone }
}

const fmtDistance = v => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : '—')
const fmtPercent = v => (typeof v === 'number' && Number.isFinite(v) ? `${Math.round(v * 100)}%` : '0%')

/**
 * Ask who this is.
 *
 * The screen has two states and shows one at a time: a question, then an
 * answer. Keeping the query form on screen next to the result was what made
 * the old layout feel like a control panel rather than a tool that replies.
 */
export default function Identify({ people, notify, onEnrolThisFace }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [threshold, setThreshold] = useState(RECOMMENDED)
  const [dragging, setDragging] = useState(false)
  const [cameraOpen, setCameraOpen] = useState(false)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const inputRef = useRef(null)
  const risk = describeRisk(threshold)

  useEffect(() => {
    if (!file) { setPreview(null); return undefined }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const accept = (f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) return notify('That file is not an image.', 'error')
    setResult(null)
    setFile(f)
  }

  const run = async () => {
    if (!file) return
    setRunning(true)
    const body = new FormData()
    body.append('image', file)
    body.append('threshold', threshold.toFixed(2))
    try {
      const res = await fetch('/identify', {
        method: 'POST', headers: { Accept: 'application/json' }, body,
      })
      const data = await res.json().catch(() => null)
      if (!res.ok || !data || data.error) {
        notify(data?.error ?? 'Identifying failed.', 'error')
        return
      }
      setResult(data)
    } catch (err) {
      notify(`Could not reach the server: ${err.message}`, 'error')
    } finally {
      setRunning(false)
    }
  }

  const reset = () => { setResult(null); setFile(null) }

  // ------------------------------------------------------------- answer
  if (result) {
    const faces = result.faces ?? []
    const known = faces.filter(f => f.is_known)
    const nobodyEnrolled = result.enrolled_count === 0

    return (
      <div className="view">
        <header className="view-head">
          <div>
            <h1 className="view-title">
              {faces.length === 0
                ? 'No face found'
                : known.length > 0
                  ? known.length === 1 ? `That's ${known[0].name}` : `${known.length} people recognised`
                  : 'Not recognised'}
            </h1>
            <p className="view-sub">
              {faces.length === 0
                ? 'Nothing in this photo looked like a face.'
                : `${faces.length === 1 ? '1 face' : `${faces.length} faces`} · strictness ${result.threshold_used}`}
            </p>
          </div>
          <div className="view-actions">
            <button type="button" className="btn btn-ghost" onClick={reset}>Try another photo</button>
          </div>
        </header>

        {result.annotated_image && faces.length > 0 && (
          <img className="answer-image" src={result.annotated_image} alt="The photo with each face outlined and labelled" />
        )}

        {faces.length === 0 && (
          <div className="note">
            Try a clearer, better-lit photo where the face is larger in frame.
          </div>
        )}

        {nobodyEnrolled && faces.length > 0 && (
          <div className="note">
            A face was found, but nobody is enrolled to compare it against.{' '}
            <button type="button" className="btn-link" onClick={() => navigate(ROUTES.ADD)}>
              Add someone first
            </button>
          </div>
        )}

        {faces.map((face, i) => (
          <article key={i} className={`answer${face.is_known ? '' : ' is-unknown'}`}>
            <div className="answer-head">
              <h2 className="answer-name">{face.is_known ? face.name : 'Unknown face'}</h2>
              {!face.no_candidates && (
                <span className="answer-confidence">{fmtPercent(face.confidence)} confident</span>
              )}
            </div>

            {!face.no_candidates && (
              <p className="answer-meta">
                distance {fmtDistance(face.distance)} · similarity {fmtPercent(face.similarity)}
              </p>
            )}

            {face.liveness?.suspicious && (
              <p className="answer-spoof">
                This may be a photo of a photo rather than a live person
                (liveness {fmtPercent(face.liveness.live_score)}). Advisory only.
              </p>
            )}

            {face.candidates?.length > 0 && (
              <Reasoning candidates={face.candidates} threshold={face.threshold_used ?? threshold} />
            )}

            {!face.is_known && !nobodyEnrolled && (
              <div className="answer-next">
                <p>Should the system know this person?</p>
                <button type="button" className="btn" onClick={() => onEnrolThisFace(file)}>
                  Enrol this face
                </button>
              </div>
            )}
          </article>
        ))}
      </div>
    )
  }

  // ------------------------------------------------------------ question
  return (
    <div className="view">
      <header className="view-head">
        <div>
          <h1 className="view-title">Identify</h1>
          <p className="view-sub">
            {people.length === 0
              ? 'Nobody is enrolled yet, so every face will come back unknown.'
              : `Compared against ${people.length === 1 ? '1 enrolled person' : `${people.length} enrolled people`}.`}
          </p>
        </div>
      </header>

      {people.length === 0 && (
        <div className="note">
          <button type="button" className="btn-link" onClick={() => navigate(ROUTES.ADD)}>
            Add someone first
          </button>{' '}
          so there is something to compare against.
        </div>
      )}

      {cameraOpen ? (
        <CameraCapture
          disabled={running}
          onCapture={f => { setCameraOpen(false); accept(f) }}
          onError={msg => notify(msg, 'error')}
          onClose={() => setCameraOpen(false)}
        />
      ) : (
        <>
          <button
            type="button"
            className={`drop drop-lg${dragging ? ' is-dragging' : ''}`}
            disabled={running}
            onClick={() => inputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => {
              e.preventDefault(); setDragging(false)
              if (!running) accept(e.dataTransfer.files[0])
            }}
          >
            <span className="drop-title">{file ? file.name : 'Drop a photo here'}</span>
            <span className="drop-sub">{file ? 'Click to choose a different one' : 'or click to browse'}</span>
          </button>
          <button type="button" className="btn-link" onClick={() => setCameraOpen(true)} disabled={running}>
            Use the camera instead
          </button>
        </>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={e => { accept(e.target.files[0]); e.target.value = '' }}
      />

      {file && preview && <img className="preview" src={preview} alt={file.name} />}

      <details className="tuning">
        <summary className="tuning-summary">Match strictness · {threshold.toFixed(2)}</summary>
        <div className="tuning-body">
          <input
            className="range"
            type="range"
            min="0.30" max="0.90" step="0.01"
            value={threshold}
            disabled={running}
            onChange={e => setThreshold(parseFloat(e.target.value))}
          />
          <p className={`risk risk-${risk.tone}`}>{risk.text}</p>
          {Math.abs(threshold - RECOMMENDED) > 0.005 && (
            <button type="button" className="btn-link" onClick={() => setThreshold(RECOMMENDED)}>
              Reset to the recommended {RECOMMENDED.toFixed(2)}
            </button>
          )}
        </div>
      </details>

      <div className="flow-actions">
        <button type="button" className="btn btn-lg" onClick={run} disabled={running || !file}>
          {running ? (<><span className="spinner" aria-hidden="true" />Looking…</>) : 'Who is this?'}
        </button>
      </div>
    </div>
  )
}

/**
 * Every enrolled person the face was compared against, nearest first.
 * The gap between first and second place is what says whether to trust it.
 */
function Reasoning({ candidates, threshold }) {
  const widest = Math.max(threshold * 1.6, ...candidates.map(c => c.distance ?? 0))
  const pct = d => `${Math.min(100, ((d ?? 0) / widest) * 100)}%`

  return (
    <details className="why">
      <summary className="why-summary">
        Why{candidates.length > 1 ? ` · ${candidates.length} compared` : ''}
      </summary>
      <div className="why-body">
        <div className="why-scale">
          <span className="why-tick-label" style={{ left: pct(threshold) }}>{threshold} threshold</span>
        </div>
        {candidates.map(c => (
          <div key={c.name} className="why-row">
            <span className="why-name" title={c.name}>{c.name}</span>
            <span className="why-bar">
              <span className={`why-fill${c.accepted ? ' is-accepted' : ''}`} style={{ width: pct(c.distance) }} />
              <span className="why-threshold" style={{ left: pct(threshold) }} aria-hidden="true" />
            </span>
            <span className="why-distance">{fmtDistance(c.distance)}</span>
          </div>
        ))}
        <p className="why-note">Lower is closer. A name is accepted only if its bar ends left of the line.</p>
      </div>
    </details>
  )
}
