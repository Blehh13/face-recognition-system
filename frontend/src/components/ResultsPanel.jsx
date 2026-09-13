function formatDistance(value) {
  // The server sends null when there was nobody to compare against.
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return value.toFixed(3)
}

function formatPercent(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '0%'
  return `${Math.round(value * 100)}%`
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
              <div className="result-meta">
                distance {formatDistance(face.distance)}
                {' · '}
                similarity {formatPercent(face.similarity)}
                {' · '}
                confidence {formatPercent(face.confidence)}
              </div>
            )}
          </div>
        ))
      )}
    </section>
  )
}
