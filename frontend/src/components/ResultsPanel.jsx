function formatDistance(value) {
  // The server sends null when there was nobody to compare against.
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return value.toFixed(3)
}

function formatPercent(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '0%'
  return `${Math.round(value * 100)}%`
}

/*
 * Shows every enrolled person the face was compared against, nearest first,
 * with the accept threshold marked.
 *
 * The point is the gap between first and second place. A match at 0.41 whose
 * runner-up is 0.43 is close to a coin toss; the same match with the runner-up
 * at 0.85 is decisive. The single winning name hides that distinction, and it
 * is exactly what someone doubting a result needs to see.
 */
function Candidates({ candidates, threshold }) {
  if (!candidates || candidates.length < 1) return null

  // Scale the bars so the threshold always sits at a consistent spot,
  // otherwise a distant outlier squashes the interesting range flat.
  const widest = Math.max(threshold * 1.6, ...candidates.map(c => c.distance ?? 0))
  const pct = d => `${Math.min(100, ((d ?? 0) / widest) * 100)}%`

  return (
    <details className="why">
      <summary className="why-summary">
        Why this result{candidates.length > 1 ? ` · ${candidates.length} compared` : ''}
      </summary>
      <div className="why-body">
        <div className="why-scale">
          <span className="why-tick" style={{ left: pct(threshold) }} aria-hidden="true" />
          <span className="why-tick-label" style={{ left: pct(threshold) }}>
            {threshold} threshold
          </span>
        </div>
        {candidates.map(c => (
          <div key={c.name} className="why-row">
            <span className="why-name" title={c.name}>{c.name}</span>
            <span className="why-bar">
              <span
                className={`why-fill${c.accepted ? ' is-accepted' : ''}`}
                style={{ width: pct(c.distance) }}
              />
              <span className="why-threshold" style={{ left: pct(threshold) }} aria-hidden="true" />
            </span>
            <span className="why-distance">{formatDistance(c.distance)}</span>
          </div>
        ))}
        <p className="why-note">
          Lower is closer. A name is accepted only if its bar ends left of the
          threshold line.
        </p>
      </div>
    </details>
  )
}

export default function ResultsPanel({ data }) {
  if (!data) {
    return (
      <section className="card">
        <div className="card-head">
          <h2 className="card-title">Results</h2>
        </div>
        <p className="empty">Choose a photo above and press Identify to see who is in it.</p>
      </section>
    )
  }

  const { faces = [], annotated_image: annotated, num_faces: count = 0, threshold_used: threshold } = data

  return (
    <section className="card">
      <div className="card-head">
        <h2 className="card-title">Results</h2>
        <span className="card-count">
          {count === 1 ? '1 face' : `${count} faces`} &middot; strictness {threshold}
        </span>
      </div>

      {annotated && count > 0 && (
        <img className="result-image" src={annotated} alt="The photo with each detected face outlined and labelled" />
      )}

      {count === 0 ? (
        <p className="empty">No faces found. Try a clearer, well-lit photo.</p>
      ) : (
        faces.map((face, index) => (
          <div
            key={`${face.name}-${index}`}
            className={`result${face.is_known ? '' : ' is-unknown'}`}
          >
            <div className="result-name">
              {face.is_known ? face.name : 'Not recognised'}
            </div>
            {face.no_candidates ? (
              <div className="result-meta">
                A face was found, but nobody is enrolled to compare it against.
              </div>
            ) : (
              <>
                <div className="result-meta">
                  distance {formatDistance(face.distance)}
                  {' · '}
                  similarity {formatPercent(face.similarity)}
                  {' · '}
                  confidence {formatPercent(face.confidence)}
                </div>
                <Candidates
                  candidates={face.candidates}
                  threshold={face.threshold_used ?? threshold}
                />
              </>
            )}
          </div>
        ))
      )}
    </section>
  )
}
