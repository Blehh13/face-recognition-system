import { useState, useRef, useEffect } from 'react'
import CameraCapture from '../components/CameraCapture.jsx'
import { ROUTES, navigate } from '../router.js'

/*
 * The threshold, its measured error curve and the slider's range all come from
 * the server (/api/config), because all three are engine-specific. Carrying a
 * copy here meant the UI advertised dlib's 0.60 while the SFace engine was
 * actually gating at 1.012 — the slider could not even reach the real value.
 *
 * These fallbacks only apply if /api/config cannot be reached.
 */
const FALLBACK_THRESHOLD = 1.012
const FALLBACK_CURVE = [[0.5, 0, 0.96], [1.0, 0, 0.05], [1.25, 0.73, 0.01], [1.5, 1, 0]]

function ratesAt(curve, t) {
  if (!curve || curve.length === 0) return { far: 0, frr: 0 }
  if (t <= curve[0][0]) return { far: curve[0][1], frr: curve[0][2] }
  const last = curve[curve.length - 1]
  if (t >= last[0]) return { far: last[1], frr: last[2] }
  for (let i = 0; i < curve.length - 1; i++) {
    const [t0, f0, r0] = curve[i]
    const [t1, f1, r1] = curve[i + 1]
    if (t >= t0 && t <= t1) {
      const k = (t1 - t0) === 0 ? 0 : (t - t0) / (t1 - t0)
      return { far: f0 + k * (f1 - f0), frr: r0 + k * (r1 - r0) }
    }
  }
  return { far: last[1], frr: last[2] }
}

function describeRisk(curve, t) {
  const { far, frr } = ratesAt(curve, t)
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
export default function Identify({ people, notify, config, onEnrolThisFace }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [threshold, setThreshold] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [cameraOpen, setCameraOpen] = useState(false)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const inputRef = useRef(null)
  const recommended = config?.threshold ?? FALLBACK_THRESHOLD
  const curve = config?.error_curve?.length ? config.error_curve : FALLBACK_CURVE
  // Span the measured range, so the control can always reach the real gate.
  const sliderMin = Math.max(0.05, Math.min(curve[0][0], recommended * 0.5))
  const sliderMax = Math.max(curve[curve.length - 1][0], recommended * 1.5)
  const active = threshold ?? recommended
  const risk = describeRisk(curve, active)

  // Adopt the server's threshold once it arrives, unless the user has moved it.
  useEffect(() => {
    if (threshold === null && config?.threshold != null) setThreshold(config.threshold)
  }, [config, threshold])

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
    body.append('threshold', active.toFixed(3))
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
              <Reasoning candidates={face.candidates} threshold={face.threshold_used ?? active} />
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
        <summary className="tuning-summary">Match strictness · {active.toFixed(3)}</summary>
        <div className="tuning-body">
          <input
            className="range"
            type="range"
            min={sliderMin} max={sliderMax} step="0.005"
            value={active}
            disabled={running}
            onChange={e => setThreshold(parseFloat(e.target.value))}
          />
          <p className={`risk risk-${risk.tone}`}>{risk.text}</p>
          {Math.abs(active - recommended) > 0.005 && (
            <button type="button" className="btn-link" onClick={() => setThreshold(recommended)}>
              Reset to the recommended {recommended.toFixed(3)}
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
