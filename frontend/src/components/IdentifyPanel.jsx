import { useState, useRef, useEffect } from 'react'

export default function IdentifyPanel({ onIdentified, onError }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [threshold, setThreshold] = useState(0.6)
  const [dragging, setDragging] = useState(false)
  const [running, setRunning] = useState(false)
  const [inlineError, setInlineError] = useState(null)
  const inputRef = useRef(null)

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
        <p className="upload-hint">
          Lower is stricter: fewer matches, but more certain ones.
        </p>
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
